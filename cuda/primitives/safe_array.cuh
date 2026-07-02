/**
 *  @brief Exception-free allocator-aware resizable array — the primitives cell's `std::vector` replacement.
 *  @file cuda/primitives/safe_array.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  @ref safe_array grows like `std::vector` but can never throw: every fallible operation is a @c try_*
 *  method returning a @ref status_t , copies are deleted (an accidental deep copy is an allocation the
 *  caller never sees fail), and moves are @c noexcept by construction. Same contract as StringZilla's
 *  @c sz::safe_vector , restated on this cell's status vocabulary.
 *
 *  Unlike `std::vector`, sizing doesn't imply filling: the explicit @c try_resize_uninitialized commits a
 *  new tail without a memset or a constructor loop, so a buffer can be sized on the host and written first
 *  by a GPU kernel — the shape @c gpu_alloc -backed device arrays rely on, since the host must never touch
 *  their bytes. The allocator must never throw — failure is a @c nullptr return, surfaced as
 *  @ref status_t::bad_alloc_k .
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <memory>      // `std::allocator_traits`, `std::allocator`
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_move_constructible_v`
#include <utility>     // `std::move`, `std::exchange`

#include "types.cuh" // `status_t`

namespace unum::uform::primitives {

/**
 *  @brief Safer alternative to `std::vector`: no exceptions, no copy constructors, and @c try_reserve /
 *      @c try_push_back / @c try_resize / @c try_assign that surface allocation failures as @ref status_t .
 *
 *  @tparam value_type_ Element type; must be nothrow-move-constructible so relocation on growth cannot
 *      throw mid-move and leave the array half-relocated.
 *  @tparam allocator_type_ Must never throw — failure is a @c nullptr return, surfaced as
 *      @ref status_t::bad_alloc_k ; must propagate on move assignment so the move assignment stays @c noexcept .
 */
template <typename value_type_, typename allocator_type_ = std::allocator<value_type_>>
class safe_array {
  public:
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocator_type = allocator_type_;

    using allocator_traits = std::allocator_traits<allocator_type>;
    using allocated_type = typename allocator_traits::value_type;
    static_assert(sizeof(value_type) == sizeof(allocated_type),
                  "Allocator value type must be the same size as the array value type");
    static_assert(allocator_traits::propagate_on_container_move_assignment::value,
                  "Allocator must propagate on move assignment, otherwise the move assignment won't be `noexcept`");
    static_assert(std::is_nothrow_move_constructible_v<value_type_>,
                  "Elements must be nothrow-move-constructible so growth relocation cannot fail mid-move");

  private:
    value_type *data_ = nullptr;
    size_type size_ = 0;
    size_type capacity_ = 0;
    [[no_unique_address]] allocator_type alloc_ {};

  public:
    safe_array() noexcept = default;
    explicit safe_array(allocator_type alloc) noexcept : alloc_(std::move(alloc)) {}
    ~safe_array() noexcept { reset(); }

    /** @warning Use @ref try_assign instead, to handle out-of-memory failures. */
    safe_array(safe_array const &) = delete;
    /** @warning Use @ref try_assign instead, to handle out-of-memory failures. */
    safe_array &operator=(safe_array const &) = delete;

    safe_array(safe_array &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)), alloc_(std::move(other.alloc_)) {}

    safe_array &operator=(safe_array &&other) noexcept {
        if (this == &other) return *this;
        reset();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        capacity_ = std::exchange(other.capacity_, 0);
        alloc_ = std::move(other.alloc_);
        return *this;
    }

    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] value_type *data() noexcept { return data_; }
    [[nodiscard]] value_type const *data() const noexcept { return data_; }
    [[nodiscard]] value_type *begin() noexcept { return data_; }
    [[nodiscard]] value_type const *begin() const noexcept { return data_; }
    [[nodiscard]] value_type *end() noexcept { return data_ + size_; }
    [[nodiscard]] value_type const *end() const noexcept { return data_ + size_; }
    [[nodiscard]] value_type &operator[](size_type index) noexcept { return data_[index]; }
    [[nodiscard]] value_type const &operator[](size_type index) const noexcept { return data_[index]; }

    /** @brief The whole live range as a read-only `std::span` — the hand-off shape for bulk consumers. */
    [[nodiscard]] std::span<value_type const> view() const noexcept { return {data_, size_}; }

    /** @brief Destroy every element; capacity is retained for reuse. */
    void clear() noexcept {
        if constexpr (!std::is_trivially_destructible_v<value_type>)
            for (size_type index = 0; index != size_; ++index) data_[index].~value_type();
        size_ = 0;
    }

    /** @brief Destroy every element and return the backing memory to the allocator. */
    void reset() noexcept {
        clear();
        if (data_ != nullptr)
            allocator_traits::deallocate(alloc_, reinterpret_cast<allocated_type *>(data_), capacity_);
        data_ = nullptr;
        capacity_ = 0;
    }

    /** @brief Grow the backing to at least @p new_capacity elements, relocating the live prefix with
     *      nothrow moves. Never shrinks.
     *  @retval status_t::bad_alloc_k The allocator returned null; the array is untouched. */
    [[nodiscard]] status_t try_reserve(size_type new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;
        allocated_type *raw = allocator_traits::allocate(alloc_, new_capacity);
        if (raw == nullptr) return status_t::bad_alloc_k;
        value_type *new_data = reinterpret_cast<value_type *>(raw);
        for (size_type index = 0; index != size_; ++index) {
            new (new_data + index) value_type(std::move(data_[index]));
            if constexpr (!std::is_trivially_destructible_v<value_type>) data_[index].~value_type();
        }
        if (data_ != nullptr)
            allocator_traits::deallocate(alloc_, reinterpret_cast<allocated_type *>(data_), capacity_);
        data_ = new_data;
        capacity_ = new_capacity;
        return status_t::success_k;
    }

    /** @brief Append @p value , doubling the capacity when full.
     *  @retval status_t::bad_alloc_k Growth failed; the array is untouched and @p value is dropped. */
    [[nodiscard]] status_t try_push_back(value_type value) noexcept {
        if (size_ == capacity_) {
            if (status_t status = try_reserve(capacity_ != 0 ? capacity_ * 2u : 1u); status != status_t::success_k)
                return status;
        }
        new (data_ + size_) value_type(std::move(value));
        ++size_;
        return status_t::success_k;
    }

    /** @brief Resize to exactly @p new_size : a shrink destroys the tail in place; a growth reserves
     *      exactly (no doubling — a resize states its target) and value-initializes the new tail. */
    [[nodiscard]] status_t try_resize(size_type new_size) noexcept {
        if (new_size <= size_) {
            if constexpr (!std::is_trivially_destructible_v<value_type>)
                for (size_type index = new_size; index != size_; ++index) data_[index].~value_type();
            size_ = new_size;
            return status_t::success_k;
        }
        if (status_t status = try_reserve(new_size); status != status_t::success_k) return status;
        for (size_type index = size_; index != new_size; ++index) new (data_ + index) value_type {};
        size_ = new_size;
        return status_t::success_k;
    }

    /** @brief @ref try_resize without touching the new tail — no memset, no constructors — so the first
     *      writer can be a GPU kernel and a device-resident buffer is never touched from the host. Only
     *      trivially-constructible elements qualify; a shrink behaves exactly like @ref try_resize . */
    [[nodiscard]] status_t try_resize_uninitialized(size_type new_size) noexcept
        requires(std::is_trivially_default_constructible_v<value_type_>)
    {
        if (new_size <= size_) return try_resize(new_size);
        if (status_t status = try_reserve(new_size); status != status_t::success_k) return status;
        size_ = new_size;
        return status_t::success_k;
    }

    /** @brief Replace the contents with a copy of @p values , sized exactly; the copy-constructor
     *      replacement for the deleted copies. */
    [[nodiscard]] status_t try_assign(std::span<value_type const> values) noexcept {
        clear();
        if (status_t status = try_reserve(values.size()); status != status_t::success_k) return status;
        for (size_type index = 0; index != values.size(); ++index) new (data_ + index) value_type(values[index]);
        size_ = values.size();
        return status_t::success_k;
    }
};

} // namespace unum::uform::primitives

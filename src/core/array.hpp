/**
 *  @brief Exception-free allocator-aware resizable array — the primitives cell's replacement for the standard vector.
 *  @file src/core/array.hpp
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  @ref array grows like the standard vector but can never throw: every fallible operation returns a
 *  @c [[nodiscard]] @ref status_t , copies are deleted (an accidental deep copy is an allocation the
 *  caller never sees fail), and moves are @c noexcept by construction. Same contract as StringZilla's
 *  @c sz::vector , restated on this cell's status vocabulary.
 *
 *  Unlike the standard vector, sizing doesn't imply filling: the explicit @c resize_uninitialized commits a
 *  new tail without a memset or a constructor loop, so a buffer can be sized on the host and written first
 *  by a GPU kernel — the shape @c gpu_alloc -backed device arrays rely on, since the host must never touch
 *  their bytes. The allocator must never throw — failure is a @c nullptr return, surfaced as
 *  @ref status_t::bad_alloc_k .
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <utility>     // `std::move`, `std::exchange`
#include <type_traits> // `std::is_nothrow_move_constructible_v`
#include <new>         // placement `new`
#include <memory>      // `std::allocator_traits`, `std::allocator`
#include <span>        // `std::span`

#include "core/types.hpp" // `status_t`

namespace unum::uform::core {

/**
 *  @brief Safer alternative to the standard vector: no exceptions, no copy constructors, and @c reserve /
 *      @c push_back / @c resize / @c assign that surface allocation failures as @ref status_t .
 *
 *  @tparam value_type_ Element type; must be nothrow-move-constructible so relocation on growth cannot
 *      throw mid-move and leave the array half-relocated.
 *  @tparam alloc_type_ Must never throw — failure is a @c nullptr return, surfaced as
 *      @ref status_t::bad_alloc_k ; must propagate on move assignment so the move assignment stays @c noexcept .
 */
template <typename value_type_, typename alloc_type_ = std::allocator<value_type_>>
class array {
  public:
    using value_type = value_type_;
    using size_type = std::size_t;
    using alloc_type = alloc_type_;

    using alloc_traits = std::allocator_traits<alloc_type>;
    static_assert(std::is_same_v<typename alloc_traits::value_type, value_type>,
                  "Allocator value type must be the array value type");
    static_assert(alloc_traits::propagate_on_container_move_assignment::value,
                  "Allocator must propagate on move assignment, otherwise the move assignment won't be `noexcept`");
    static_assert(std::is_nothrow_move_constructible_v<value_type_>,
                  "Elements must be nothrow-move-constructible so growth relocation cannot fail mid-move");

  private:
    value_type *data_ = nullptr;
    size_type size_ = 0;
    size_type capacity_ = 0;
    [[no_unique_address]] alloc_type alloc_ {};

  public:
    array() noexcept = default;
    explicit array(alloc_type alloc) noexcept : alloc_(std::move(alloc)) {}
    ~array() noexcept { reset(); }

    /** @warning Use @ref assign instead, to handle out-of-memory failures. */
    array(array const &) = delete;

    /** @warning Use @ref assign instead, to handle out-of-memory failures. */
    array &operator=(array const &) = delete;

    array(array &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)), alloc_(std::move(other.alloc_)) {}

    array &operator=(array &&other) noexcept {
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

    /** The whole live range as a read-only `std::span` — the hand-off shape for bulk consumers. */
    [[nodiscard]] std::span<value_type const> view() const noexcept { return {data_, size_}; }

    /** Destroy every element; capacity is retained for reuse. */
    void clear() noexcept {
        if constexpr (!std::is_trivially_destructible_v<value_type>)
            for (size_type index = 0; index != size_; ++index) data_[index].~value_type();
        size_ = 0;
    }

    /** Destroy every element and return the backing memory to the allocator. */
    void reset() noexcept {
        clear();
        if (data_ != nullptr) alloc_traits::deallocate(alloc_, data_, capacity_);
        data_ = nullptr;
        capacity_ = 0;
    }

    /**
     *  @brief Grow the backing to at least @p new_capacity elements, relocating the live prefix with
     *      nothrow moves. Never shrinks.
     *  @retval status_t::bad_alloc_k The allocator returned null; the array is untouched.
     */
    [[nodiscard]] status_t reserve(size_type new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;
        value_type *new_data = alloc_traits::allocate(alloc_, new_capacity);
        if (new_data == nullptr) return status_t::bad_alloc_k;
        for (size_type index = 0; index != size_; ++index) {
            new (new_data + index) value_type(std::move(data_[index]));
            if constexpr (!std::is_trivially_destructible_v<value_type>) data_[index].~value_type();
        }
        if (data_ != nullptr) alloc_traits::deallocate(alloc_, data_, capacity_);
        data_ = new_data;
        capacity_ = new_capacity;
        return status_t::success_k;
    }

    /**
     *  @brief Append @p value , doubling the capacity when full.
     *  @retval status_t::bad_alloc_k Growth failed; the array is untouched and @p value is dropped.
     */
    [[nodiscard]] status_t push_back(value_type value) noexcept {
        if (size_ == capacity_) {
            if (status_t status = reserve(capacity_ != 0 ? capacity_ * 2u : 1u); status != status_t::success_k)
                return status;
        }
        new (data_ + size_) value_type(std::move(value));
        ++size_;
        return status_t::success_k;
    }

    /** Resize to exactly @p new_size : a shrink destroys the tail in place; a growth reserves
     *  exactly (no doubling — a resize states its target) and value-initializes the new tail. */
    [[nodiscard]] status_t resize(size_type new_size) noexcept {
        if (new_size <= size_) {
            if constexpr (!std::is_trivially_destructible_v<value_type>)
                for (size_type index = new_size; index != size_; ++index) data_[index].~value_type();
            size_ = new_size;
            return status_t::success_k;
        }
        if (status_t status = reserve(new_size); status != status_t::success_k) return status;
        for (size_type index = size_; index != new_size; ++index) new (data_ + index) value_type {};
        size_ = new_size;
        return status_t::success_k;
    }

    /** @ref resize without touching the new tail — no memset, no constructors — so the first
     *  writer can be a GPU kernel and a device-resident buffer is never touched from the host. Only
     *  trivially-constructible elements qualify; a shrink behaves exactly like @ref resize . */
    [[nodiscard]] status_t resize_uninitialized(size_type new_size) noexcept
        requires(std::is_trivially_default_constructible_v<value_type_>)
    {
        if (new_size <= size_) return resize(new_size);
        if (status_t status = reserve(new_size); status != status_t::success_k) return status;
        size_ = new_size;
        return status_t::success_k;
    }

    /** Replace the contents with a copy of @p values , sized exactly; the copy-constructor
     *  replacement for the deleted copies. */
    [[nodiscard]] status_t assign(std::span<value_type const> values) noexcept {
        clear();
        if (status_t status = reserve(values.size()); status != status_t::success_k) return status;
        for (size_type index = 0; index != values.size(); ++index) new (data_ + index) value_type(values[index]);
        size_ = values.size();
        return status_t::success_k;
    }
};

/**
 *  @brief Allocator-free sibling of @ref array : inline storage for up to @p capacity_limit_ elements,
 *      a runtime size, and no heap behind it — the natural owner of per-GPU object tables, where the
 *      bound is a small compile-time constant and elements must never relocate once constructed.
 *
 *  Same contract otherwise: no exceptions, no copies, @c noexcept moves (element-wise — the storage is
 *  inline), and growth that can only fail by exceeding the capacity limit.
 */
template <typename value_type_, std::size_t capacity_limit_>
class stack_array {
  public:
    using value_type = value_type_;
    using size_type = std::size_t;

    static_assert(std::is_nothrow_move_constructible_v<value_type_>,
                  "Elements must be nothrow-move-constructible so container moves stay `noexcept`");

    stack_array() noexcept {}
    stack_array(stack_array const &) = delete;
    stack_array &operator=(stack_array const &) = delete;
    stack_array(stack_array &&other) noexcept {
        for (size_type index = 0; index != other.size_; ++index)
            new (pointer_at(index)) value_type(std::move(other[index]));
        size_ = other.size_;
        other.clear();
    }
    stack_array &operator=(stack_array &&other) noexcept {
        if (this != &other) {
            clear();
            for (size_type index = 0; index != other.size_; ++index)
                new (pointer_at(index)) value_type(std::move(other[index]));
            size_ = other.size_;
            other.clear();
        }
        return *this;
    }
    ~stack_array() noexcept { clear(); }

    [[nodiscard]] static constexpr size_type capacity() noexcept { return capacity_limit_; }
    [[nodiscard]] size_type size() const noexcept { return size_; }

    [[nodiscard]] value_type &operator[](size_type index) noexcept { return *std::launder(pointer_at(index)); }
    [[nodiscard]] value_type const &operator[](size_type index) const noexcept {
        return *std::launder(pointer_at(index));
    }
    [[nodiscard]] value_type *begin() noexcept { return std::launder(pointer_at(0)); }
    [[nodiscard]] value_type const *begin() const noexcept { return std::launder(pointer_at(0)); }
    [[nodiscard]] value_type *end() noexcept { return begin() + size_; }
    [[nodiscard]] value_type const *end() const noexcept { return begin() + size_; }
    [[nodiscard]] std::span<value_type> span() noexcept { return {begin(), size_}; }
    [[nodiscard]] std::span<value_type const> view() const noexcept { return {begin(), size_}; }

    /** Move-append one element; @ref status_t::bad_alloc_k past the capacity limit. */
    [[nodiscard]] status_t push_back(value_type value) noexcept {
        if (size_ == capacity_limit_) return status_t::bad_alloc_k;
        new (pointer_at(size_)) value_type(std::move(value));
        ++size_;
        return status_t::success_k;
    }

    /** Grow to @p new_size default-constructed elements (or shrink, destroying the tail);
     *  @ref status_t::bad_alloc_k past the capacity limit. */
    [[nodiscard]] status_t resize(size_type new_size) noexcept {
        if (new_size > capacity_limit_) return status_t::bad_alloc_k;
        for (; size_ > new_size; --size_) std::launder(pointer_at(size_ - 1))->~value_type();
        for (; size_ < new_size; ++size_) new (pointer_at(size_)) value_type();
        return status_t::success_k;
    }

    void clear() noexcept {
        for (size_type index = size_; index != 0; --index) std::launder(pointer_at(index - 1))->~value_type();
        size_ = 0;
    }

  private:
    [[nodiscard]] value_type *pointer_at(size_type index) noexcept { return items_ + index; }
    [[nodiscard]] value_type const *pointer_at(size_type index) const noexcept { return items_ + index; }

    /** A union member, so no element is constructed or destroyed until @ref push_back or @ref resize says so. */
    union {
        value_type_ items_[capacity_limit_];
    };
    size_type size_ = 0;
};

/** One entry per GPU of this host_domain — the per-GPU object table every composition root
 *  owns and threads down by reference; see @ref max_gpus_per_host_k . */
template <typename value_type_>
using per_gpu_array = stack_array<value_type_, max_gpus_per_host_k>;

} // namespace unum::uform::core

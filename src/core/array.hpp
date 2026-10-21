/**
 *  @file src/core/array.hpp
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief Exception-free fixed-capacity arrays with inline storage, for the per-GPU object tables.
 *
 *  @ref stack_array holds up to a compile-time number of elements inline and can never throw: every
 *  fallible operation returns a @c [[nodiscard]] @ref status_t , copies are deleted, and moves are
 *  @c noexcept by construction. Growable arrays are @c st::basic_vector .
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <utility>     // `std::move`
#include <type_traits> // `std::is_nothrow_move_constructible_v`
#include <new>         // placement `new`, `std::launder`
#include <span>        // `std::span`

#include "core/types.hpp" // `status_t`

namespace unum::uform::core {

/**
 *  @brief Inline storage for up to @p capacity_limit_ elements, a runtime size, and no heap behind
 *      it — the natural owner of per-GPU object tables, where the bound is a small compile-time
 *      constant and elements must never relocate once constructed.
 *
 *  No exceptions, no copies, @c noexcept moves (element-wise — the storage is inline), and growth
 *  that can only fail by exceeding the capacity limit.
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

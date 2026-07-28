#include <cstddef>
#include <cstring>
#include <cstdint>

#if defined(_MSC_VER)
    #define FORCE_INLINE static __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE static inline __attribute__((always_inline))
#else
    #define FORCE_INLINE static inline
#endif

#if defined(_MSVC_LANG) && _MSVC_LANG >= 202002L
    #include <bit>
    #define NCB_HAS_BIT_CAST 1
#elif defined(__cplusplus) && __cplusplus >= 202002L && defined(__has_include)
    #if __has_include(<bit>)
        #include <bit>
        #define NCB_HAS_BIT_CAST 1
    #endif
#endif

#ifndef NCB_HAS_BIT_CAST
    #define NCB_HAS_BIT_CAST 0
#endif

namespace ncb {

    FORCE_INLINE int fast_isfinite(double a) {
    #if NCB_HAS_BIT_CAST
        uint64_t bits = std::bit_cast<uint64_t>(a);
    #else
        uint64_t bits;
        std::memcpy(&bits, &a, sizeof(double));
    #endif
        return (bits & 0x7FFFFFFFFFFFFFFFULL) < 0x7FF0000000000000ULL;
    }

    FORCE_INLINE int fast_isfinite(float a) {
    #if NCB_HAS_BIT_CAST
        uint32_t bits = std::bit_cast<uint32_t>(a);
    #else
        uint32_t bits;
        std::memcpy(&bits, &a, sizeof(float));
    #endif
        return (bits & 0x7FFFFFFF) < 0x7F800000;
    }

    template <typename T>
    FORCE_INLINE constexpr T unchecked_max(T a, T b) {
        return (a > b) ? a : b;
    }

    FORCE_INLINE constexpr size_t min_sz(size_t a, size_t b) {
        return (a < b) ? a : b;
    }

    FORCE_INLINE size_t get_start_pos(
        size_t write_head, size_t size, size_t maxlen
    ){
        return (write_head >= size)
              ? write_head - size
              : write_head + maxlen - size;
    }

    namespace append {

        FORCE_INLINE size_t append_advance_head(size_t head, size_t maxlen) {
            head++;
            return (head >= maxlen) ? 0 : head;
        }

        template <typename T>
        FORCE_INLINE int overwritten_never_append(
            T*      buf_ptr,
            size_t* write_head_ptr,
            size_t* size_ptr,
            size_t  maxlen,
            T       value
        ){
            size_t write_head = *write_head_ptr;

            if (*size_ptr < maxlen) {
                (*size_ptr)++;
            }

            buf_ptr[write_head] = value;
            *write_head_ptr = append_advance_head(write_head, maxlen);

            return 0;
        }

        template <typename T>
        FORCE_INLINE int overwritten_always_append(
            T*      buf_ptr,
            size_t* write_head_ptr,
            size_t* size_ptr,
            size_t  maxlen,
            T       value,
            T*      out_overwritten
        ){
            int overwritten;
            size_t write_head = *write_head_ptr;

            if (*size_ptr < maxlen) {
                (*size_ptr)++;
                overwritten = 0;
            }
            else {
                *out_overwritten = buf_ptr[write_head];
                overwritten = 1;
            }

            buf_ptr[write_head] = value;
            *write_head_ptr = append_advance_head(write_head, maxlen);

            return overwritten;
        }

        template <typename T>
        FORCE_INLINE int overwritten_conditional_append(
            T*      buf_ptr,
            size_t* write_head_ptr,
            size_t* size_ptr,
            size_t  maxlen,
            T       value,
            T*      out_overwritten,
            int     return_overwritten
        ){
            int overwritten;
            size_t write_head = *write_head_ptr;

            if (*size_ptr < maxlen) {
                (*size_ptr)++;
                overwritten = 0;
            }
            else {
                if (return_overwritten) {
                    *out_overwritten = buf_ptr[write_head];
                    overwritten = 1;
                }
                else {
                    overwritten = 0;
                }
            }

            buf_ptr[write_head] = value;
            *write_head_ptr = append_advance_head(write_head, maxlen);

            return overwritten;
        }
    }

    namespace extend {

        FORCE_INLINE size_t extend_advance_head(
            size_t head, size_t maxlen, size_t n
        ){
            head += n;
            return (head >= maxlen) ? head - maxlen : head;
        }

        FORCE_INLINE void write_data(
            char*       buf_ptr,
            const char* src_ptr,
            size_t      start,
            size_t      maxlen,
            size_t      n,
            size_t      elem_size
        ){
            if (start + n <= maxlen) {
                std::memcpy(
                    buf_ptr + (start * elem_size),
                    src_ptr,
                    n * elem_size
                );
            }
            else {
                const size_t p1_len = maxlen - start;
                std::memcpy(
                    buf_ptr + (start * elem_size),
                    src_ptr,
                    p1_len * elem_size
                );
                std::memcpy(
                    buf_ptr,
                    src_ptr + (p1_len * elem_size),
                    (n - p1_len) * elem_size
                );
            }
        }

        FORCE_INLINE void read_data(
            char*       dest_ptr,
            const char* buf_ptr,
            size_t      start,
            size_t      maxlen,
            size_t      n,
            size_t      elem_size
        ){
            if (start + n <= maxlen) {
                std::memcpy(
                    dest_ptr,
                    buf_ptr + (start * elem_size),
                    n * elem_size
                );
            }
            else {
                const size_t p1_len = maxlen - start;
                std::memcpy(
                    dest_ptr,
                    buf_ptr + (start * elem_size),
                    p1_len * elem_size
                );
                std::memcpy(
                    dest_ptr + (p1_len * elem_size),
                    buf_ptr,
                    (n - p1_len) * elem_size
                );
            }
        }

        FORCE_INLINE void read_overwritten(
            char*       dest_ptr,
            const char* buf_ptr,
            size_t      write_head,
            size_t      size, 
            size_t      maxlen,
            size_t      n,
            size_t      elem_size
        ){
            const size_t start = get_start_pos(write_head, size, maxlen);
            read_data(
                dest_ptr,
                buf_ptr,
                start,
                maxlen,
                n,
                elem_size
            );
        }
    }
}
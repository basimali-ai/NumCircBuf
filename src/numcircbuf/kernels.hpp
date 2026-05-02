#include <cstddef>
#include <cstring>

#if defined(_MSC_VER)
    #define FORCE_INLINE static __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE static inline __attribute__((always_inline))
#else
    #define FORCE_INLINE static inline
#endif

namespace ncb{

    namespace append {

        FORCE_INLINE size_t append_advance_head(
            size_t head, size_t maxlen
        ){
            head++;
            if (head >= maxlen) {
                head = 0;
            }
            return head;
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
            if (head >= maxlen) {
                head -= maxlen;
            }
            return head;
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
            const size_t start = (write_head >= size) 
                                ? (write_head - size) 
                                : (write_head + maxlen - size);

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
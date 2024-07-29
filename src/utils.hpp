#pragma once
#include <cuComplex.h>
#include <dlfcn.h>
#include <functional>
#include <ozimmu/ozimmu.hpp>
#include <stdexcept>
#include <unistd.h>

#define OZIMMU_NOT_IMPLEMENTED                                                 \
  mtk::ozimmu::detail::print_not_implemented(__FILE__, __LINE__, __func__)

namespace mtk {
namespace ozimmu {
namespace detail {
inline void print_not_implemented(const std::string file,
                                  const std::size_t line,
                                  const std::string func) {
  throw std::runtime_error("Not implemented (" + func + " in " + file + ", l." +
                           std::to_string(line) + ")");
}

template <class T> struct real_type {
  using type = T;
};
template <> struct real_type<cuDoubleComplex> {
  using type = double;
};
} // namespace detail

template <class slice_element_t>
inline std::uint32_t padded_ld(const std::uint32_t n) {
  using align_t = std::uint32_t;
  if (sizeof(align_t) >= sizeof(slice_element_t)) {
    const auto f = sizeof(align_t) / sizeof(slice_element_t);
    return ((n + f - 1) / f) * f;
  } else {
    return n;
  }
}

template <class slice_element_t>
inline std::uint32_t
get_slice_ld(const std::uint32_t m, const std::uint32_t n,
             const mtk::ozimmu::operation_t op,
             const mtk::ozimmu::data_t d = mtk::ozimmu::data_t::none) {
  if constexpr (!std::is_same_v<slice_element_t, void>) {
    return padded_ld<slice_element_t>(op == mtk::ozimmu::op_n ? m : n);
  } else {
    if (mtk::ozimmu::get_data_size_in_byte(d) == 0) {
      return 0;
    } else if (mtk::ozimmu::get_data_size_in_byte(d) == 1) {
      return get_slice_ld<std::uint8_t>(m, n, op);
    } else if (mtk::ozimmu::get_data_size_in_byte(d) == 2) {
      return get_slice_ld<std::uint16_t>(m, n, op);
    } else if (mtk::ozimmu::get_data_size_in_byte(d) == 4) {
      return get_slice_ld<std::uint32_t>(m, n, op);
    }
    OZIMMU_NOT_IMPLEMENTED;
  }

  OZIMMU_NOT_IMPLEMENTED;
  return 0;
}

template <class slice_element_t>
inline std::uint32_t get_slice_num_elements(
    const std::uint32_t m, const std::uint32_t n,
    const mtk::ozimmu::operation_t op,
    const mtk::ozimmu::data_t dtype = mtk::ozimmu::data_t::none) {
  return get_slice_ld<slice_element_t>(m, n, op, dtype) *
         (op == mtk::ozimmu::op_n ? n : m);
}

} // namespace ozimmu
} // namespace mtk

inline std::string
ozIMMU_load_env_if_defined(const std::string env_str,
                           const std::string default_v = "") {
  const auto env = getenv(env_str.c_str());
  if (env != nullptr) {
    return env;
  }
  return default_v;
}

// For logging
inline void ozIMMU_run_if_env_defined(const std::string env_str,
                                      const std::function<void(void)> func,
                                      const bool default_v = 0) {
  const auto env = getenv(env_str.c_str());
  if ((env != nullptr && std::string(env) != "0") ||
      (env == nullptr && default_v)) {
    func();
  }
}

inline void ozIMMU_log(const std::string str) {
  const std::string info_env_name = "OZIMMU_INFO";
  ozIMMU_run_if_env_defined(info_env_name, [&]() {
    std::fprintf(stdout, "[ozIMMU LOG] %s\n", str.c_str());
    std::fflush(stdout);
  });
}

inline void ozIMMU_error(const std::string str) {
  const std::string error_env_name = "OZIMMU_ERROR";
  ozIMMU_run_if_env_defined(
      error_env_name,
      [&]() {
        std::fprintf(stdout, "[ozIMMU ERROR] %s\n", str.c_str());
        std::fflush(stdout);
      },
      1);
}

inline void *ozIMMU_get_function_pointer(const std::string library_name,
                                         const std::string function_name) {

  // Open the library
  const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
  if (lib_ptr == nullptr) {
    ozIMMU_error("Failed to load " + library_name +
                 ". Default rule will be used.");
    return nullptr;
  }

  // Get function pointer
  void *function_ptr = dlsym(lib_ptr, function_name.c_str());
  if (function_ptr == NULL) {
    ozIMMU_log(
        "Failed to load a function " + function_name +
        " during selecting hijacking function. Default rule will be used.");
    return nullptr;
  }

  return function_ptr;
}

inline int check_gemm_shape(const mtk::ozimmu::operation_t op,
                            const std::size_t m, const std::size_t n,
                            const std::size_t ld, const std::string mat_name) {
  if ((op == mtk::ozimmu::op_n ? m : n) > ld) {
    const std::string message =
        "The leading dimension of " + mat_name + " (" + std::to_string(ld) +
        ") must be larger or equal to the number of " +
        (op == mtk::ozimmu::op_n ? "rows" : "cols") + " (" +
        std::to_string(op == mtk::ozimmu::op_n ? m : n) + ")";
    ozIMMU_error(message);
    return 1;
  }
  return 0;
}

template <class T>
inline int check_address_alignment(const T *const ptr,
                                   const std::string mat_name) {
  if (reinterpret_cast<std::uint64_t>(ptr) % sizeof(T)) {
    const std::string message =
        "Invalid address alignment for matrix " + mat_name;
    ozIMMU_error(message);
    return 1;
  }
  return 0;
}

#pragma once
#include <unistd.h>
#include <dlfcn.h>
#include <stdexcept>
#include <functional>
#include <cuComplex.h>

namespace mtk {
namespace ozimmu {
namespace detail {
inline void print_not_implemented(
		const std::string file,
		const std::size_t line,
		const std::string func
		) {
	throw std::runtime_error("Not implemented (" + func + " in " + file + ", l." + std::to_string(line) + ")");
}

template <class T>
struct real_type {using type = T;};
template <>
struct real_type<cuDoubleComplex> {using type = double;};
} // namespace detail
} // namespace ozimmu
} // namespace mtk


// For logging
inline void ozTCECGEMM_run_if_env_defined(
		const std::string env_str,
		const std::function<void(void)> func
		) {
	const auto env = getenv(env_str.c_str());
	if (env != nullptr && std::string(env) != "0") {
		func();
	}
}

inline void ozIMMU_log(
		const std::string str
		) {
	const std::string info_env_name = "OZIMMU_INFO";
	ozTCECGEMM_run_if_env_defined(
			info_env_name,
			[&](){
				std::fprintf(stdout, "[ozIMMU LOG] %s\n",
						str.c_str());
				std::fflush(stdout);
			});
}

inline void ozIMMU_error(
		const std::string str
		) {
	const std::string error_env_name = "OZIMMU_ERROR_LOG";
	ozTCECGEMM_run_if_env_defined(
			error_env_name,
			[&](){
				std::fprintf(stdout, "[ozIMMU ERROR] %s\n",
						str.c_str());
				std::fflush(stdout);
			});
}

inline void* ozIMMU_get_function_pointer(const std::string library_name, const std::string function_name) {

	// Open the library
	const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
	if (lib_ptr == nullptr) {
		ozIMMU_error("Failed to load " + library_name + ". Default rule will be used.");
		return nullptr;
	}

	// Get function pointer
	void* function_ptr = dlsym(lib_ptr, function_name.c_str());
	if (function_ptr == NULL) {
		ozIMMU_log("Failed to load a function " + function_name + " during selecting hijacking function. Default rule will be used.");
		return nullptr;
	}

	return function_ptr;
}

#define OZIMMU_NOT_IMPLEMENTED mtk::ozimmu::detail::print_not_implemented(__FILE__, __LINE__, __func__)

#pragma once
#include <oztcecgemm/oztcecgemm.hpp>
#include <shgemm/shgemm.hpp>
#include <cutf/cublas.hpp>

struct mtk::oztcecgemm::handle_t {
	cublasHandle_t cublas_handle;
	mtk::shgemm::shgemmHandle_t shgemm_handle;
};

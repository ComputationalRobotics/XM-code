
function(add_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)

pybind11_add_module(${EXAMPLE_NAME} ${EXAMPLE_SOURCES})
# set_property(TARGET ${EXAMPLE_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(${EXAMPLE_NAME}
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_compile_definitions(${EXAMPLE_NAME}
    PUBLIC
        EIGEN_NO_CUDA
        EIGEN_DONT_VECTORIZE
        EIGEN_DONT_ALIGN
        $<$<COMPILE_LANGUAGE:CUDA>:EIGEN_NO_CUDA;EIGEN_DONT_VECTORIZE;EIGEN_DONT_ALIGN>
)
target_link_libraries(${EXAMPLE_NAME}
    PUBLIC
        cusolver
        cublas
        cublasLt
        cusparse
        cusolverMg
        
)

install(
    TARGETS ${EXAMPLE_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)

endfunction()


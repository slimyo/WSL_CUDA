# 创建可执行文件 module_a
add_executable(hello hello.cpp)

# 链接公共库（如果有的话）
# target_link_libraries(hello common_lib)
# target_include_directories(mylib
#     PUBLIC        # 使用此库的用户也需要这些头文件
#         ${CMAKE_CURRENT_SOURCE_DIR}

    # PRIVATE       # 仅内部实现需要
    #     ${CMAKE_CURRENT_SOURCE_DIR}/internal

    # INTERFACE     # 仅使用者需要，库自身不需要
    #     ${CMAKE_SOURCE_DIR}/include/api
# )

# 设置输出目录（可选）
set_target_properties(hello PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
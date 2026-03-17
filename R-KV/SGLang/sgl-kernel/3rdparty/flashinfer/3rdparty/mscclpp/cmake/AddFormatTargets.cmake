# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Add targets to run clang-format and black

add_custom_target(check-format)
add_custom_target(format)

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    message(STATUS "Found clang-format: ${CLANG_FORMAT}")
    set(FIND_DIRS ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/python ${PROJECT_SOURCE_DIR}/test)
    add_custom_target(check-format-cpp ALL
        COMMAND ${CLANG_FORMAT} -style=file --dry-run `find ${FIND_DIRS} -type f -name *.h -o -name *.hpp -o -name *.c -o -name *.cc -o -name *.cpp -o -name *.cu`
    )
    add_dependencies(check-format check-format-cpp)
    add_custom_target(format-cpp
        COMMAND ${CLANG_FORMAT} -style=file -i `find ${FIND_DIRS} -type f -name *.h -o -name *.hpp -o -name *.c -o -name *.cc -o -name *.cpp -o -name *.cu`
    )
    add_dependencies(format format-cpp)
else()
    message(STATUS "clang-format not found.")
endif()

find_program(BLACK black)
if (BLACK)
    message(STATUS "Found black: ${BLACK}")
    add_custom_target(check-format-py
        COMMAND ${BLACK} --config ${PROJECT_SOURCE_DIR}/pyproject.toml --check ${PROJECT_SOURCE_DIR}
    )
    add_dependencies(check-format check-format-py)
    add_custom_target(format-py
        COMMAND ${BLACK} --config ${PROJECT_SOURCE_DIR}/pyproject.toml ${PROJECT_SOURCE_DIR}
    )
    add_dependencies(format format-py)
else()
    message(STATUS, "black not found.")
endif()

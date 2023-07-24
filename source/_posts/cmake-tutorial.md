---
title: cmake_tutorial
date: 2023-07-24 09:46:33
tags: CMake
categories: Knowledge
---

# CMake Note

官网：[CMake](https://cmake.org/)

官方教材：[Step 1: A Basic Starting Point — CMake 3.25.1 Documentation](https://cmake.org/cmake/help/latest/guide/tutorial/A Basic Starting Point.html#exercise-1-building-a-basic-project)

tutorial文件：cmake-3.25.1-tutorial-source

## 目录

[toc]

## Build and Run

创建build文件夹

`mkdir build`

进入build文件夹

`cd .\build\`

运行cmake配置项目并生成原生构建系统

`cmake ..`

调用构建系统真正编译链接项目

`cmake --build .`

可执行文件位于build/Debug

[toc]

## Step1:A Basic Starting Point 基础起始部分

### Exe1: Building a Basic Project 基本文件结构

最基本的CMake项目是从单个源代码文件构建的可执行文件。对于像这样的简单项目，只需要一个带有三个命令的CMakeLists.txt文件。

> **注意：**尽管 CMake 支持大写、小写和混合大小写命令，但首选小写命令，并将在整个教程中使用。

> 任何项目的最顶层 CMakeLists.txt 必须首先使用指定最低 CMake 版本[`cmake_minimum_required()`](https://cmake.org/cmake/help/latest/command/cmake_minimum_required.html#command:cmake_minimum_required)命令。

要开始一个项目，我们使用[`project()`](https://cmake.org/cmake/help/latest/command/project.html#command:project)命令设置项目名称。每个项目都需要此调用，并且应在之后立即调用 [`cmake_minimum_required()`](https://cmake.org/cmake/help/latest/command/cmake_minimum_required.html#command:cmake_minimum_required). 

最后，[`add_executable()`](https://cmake.org/cmake/help/latest/command/add_executable.html#command:add_executable)命令告诉 CMake 使用指定的源代码文件创建可执行文件

#### Helpful Resources:

- [`add_executable()`](https://cmake.org/cmake/help/latest/command/add_executable.html#command:add_executable)
- [`cmake_minimum_required()`](https://cmake.org/cmake/help/latest/command/cmake_minimum_required.html#command:cmake_minimum_required)
- [`project()`](https://cmake.org/cmake/help/latest/command/project.html#command:project)

#### Solution:

上所述，我们只需要三行CMakeLists.txt就可以启动和运行了。第一行是使用cmake_minimum_required（）设置cmake版本，如下所示：

`cmake_minimum_required(VERSION 3.10)`

使用project()命令设置项目名:

`project(Tutorial)`

添加可执行文件以及源文件：

`add_executable(Tutorial tutorial.cxx)`

### Exe2 : Specifying the C++ Standard

CMake有一些特殊变量，在幕后创建或在项目设置时对CMake有意义，以CMAKE_开头，应避免设置变量时使用，如`CMAKE_CXX_STANDARD`和`CMAKE_CXX _STANDARD_REQUIRED`可以用来指定项目所需的C++ 标准

#### Helpful Resources:

- [`CMAKE_CXX_STANDARD`](https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html#variable:CMAKE_CXX_STANDARD)
- [`CMAKE_CXX_STANDARD_REQUIRED`](https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD_REQUIRED.html#variable:CMAKE_CXX_STANDARD_REQUIRED)
- [`set()`](https://cmake.org/cmake/help/latest/command/set.html#command:set)

#### Solution:

添加一个C++11特性：  const double inputValue = std::stod(argv[1]);

remove #include<cstdlib>

设置CMAKE_CXX_STANDARD和CMAKE_CXX_STANDARD_REQUIRED，指定C++版本为11，并

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED True)

### Exe3: Adding a Version Number and Configured Header File

有时在CMakeLists.txt文件中定义的变量在源代码中也可能有用。因此，我们希望添加版本号。

一个实现方式是使用配置的头文件，我们创建一个输入文件包含多个要替换的变量。这些变量有着特殊语法`@VAR@`。然后，我们使用`configure_file()`目录复制给定的输入文件复制到一个输出文件并用CMakelists.txt中VAR的当前值替换。

#### Helpful Resources

- [`_VERSION_MAJOR`](https://cmake.org/cmake/help/latest/variable/PROJECT-NAME_VERSION_MAJOR.html#variable:_VERSION_MAJOR)
- [`_VERSION_MINOR`](https://cmake.org/cmake/help/latest/variable/PROJECT-NAME_VERSION_MINOR.html#variable:_VERSION_MINOR)
- [`configure_file()`](https://cmake.org/cmake/help/latest/command/configure_file.html#command:configure_file)
- [`target_include_directories()`](https://cmake.org/cmake/help/latest/command/target_include_directories.html#command:target_include_directories)

#### Solution:

修改CMakeLists.txt文件，使用project命令设置项目名称和版本号。

`project(Tutorial VERSION 1.0)`

使用configure_file()命令复制输入文件，并使用CMake中的变量替换

`configure_file(TutorialConfig.h.in TutorialConfig.h)`

由于配置的文件将被写入项目二进制目录，因此我们必须将该目录添加到路径列表中以搜索包含文件。

使用target_include_directories()指定可执行模板应在何处查找包含文件

```cmake
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

TutorialConfig.h.in是要配置的输入头文件。当从CMakeLists.txt调用configure_file（）时，@Tutorial_VERSION_MAJOR@和@Tutorial_VERSION_MINOR@的值将替换为TutorialConfig.h中项目的相应版本号。

```cmake
// the configured options and settings for Tutorial
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
```

修改tutorial.cxx包含头文件TutorialConfig.h

```cpp
#include "TutorialConfig.h"
```

更新tutorial.cxx打印出版本号

```cpp
  if (argc < 2) {
    // report version
    std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
              << Tutorial_VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }
```

## Step2: 添加一个链接库

### Exe1: 创建一个动态链接库

在CMake中使用add_library()命令添加链接库并指定由那些源文件组成。

我们可以用一个或多个子目录来组织项目，而不是将所有源文件放在一个目录中。我们将为库创建一个子目录。添加一个新的CMakeLists.txt文件和多个源文件。在最高级的CMakeLists.txt文件中，我们使用add_subdirectory()来将子目录添加到构建中。

一旦创建了库可以使用target_include_directories（）和target_link_libraries（）链接到我们的可执行目标文件

#### Helpful Resources

- [`add_library()`](https://cmake.org/cmake/help/latest/command/add_library.html#command:add_library)
- [`add_subdirectory()`](https://cmake.org/cmake/help/latest/command/add_subdirectory.html#command:add_subdirectory)
- [`target_include_directories()`](https://cmake.org/cmake/help/latest/command/target_include_directories.html#command:target_include_directories)
- [`target_link_libraries()`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html#command:target_link_libraries)
- [`PROJECT_SOURCE_DIR`](https://cmake.org/cmake/help/latest/variable/PROJECT_SOURCE_DIR.html#variable:PROJECT_SOURCE_DIR)

#### Solution:

文件目录

│  CMakeLists.txt
│  tutorial.cxx
│  TutorialConfig.h.in
│
└─MathFunctions		// 库文件目录
        CMakeLists.txt
        MathFunctions.h
        mysqrt.cxx

在库文件目录内的CMakeLists.txt中，创建一个目标库名为MathFunctions使用`add_library()`命令，并使用参数将源文件添加进入add_library()命令中。

```cmake
add_library(MathFunctions mysqrt.cxx)
```

之后在最顶级CMakeLists.txt使用`add_subdirectory()`命令指定次级目录。

```cmake
add_subdirectory(MathFunctions)
```

使用`target_link_libraries()`链接库文件和可执行文件

```cmake
target_link_libraries(Tutorial PUBLIC MathFunctions)
```

指定库的头文件位置，修改`target_include_directories()`添加MathFunctions次级目录为一个头文件目录使得MathFunction.h头文件可以被找到。

```cmake
target_include_directories(Tutorial PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MathFunctions"
                          )
```

源文件`tutorial.cxx`中添加头文件

```cpp
#include "MathFunctions.h"
```

调用库文件里的函数

```cpp
const double outputValue = mysqrt(inputValue);
```

### Exe2: 使我们的库具有分支特性

CMake可以使用`option`指令来完成分支性。当配置CMake文件的时候，它给予用户一个可更改的变量。这项设置江北存储在缓存中因此用户不必每次运行编译时去设置他们的值。

#### Helpful Resources

- [`if()`](https://cmake.org/cmake/help/latest/command/if.html#command:if)
- [`list()`](https://cmake.org/cmake/help/latest/command/list.html#command:list)
- [`option()`](https://cmake.org/cmake/help/latest/command/option.html#command:option)
- [`cmakedefine`](https://cmake.org/cmake/help/latest/command/configure_file.html#command:configure_file)

#### Solution:

向顶级CMakeLists.txt中添加一个选项，在gui中添加时默认启用，而CMakeLists.txt中不指定时为OFF

```cmake
option(USE_MYMATH "Use tutorial provided math implementation" ON)
```

> option 用法：option(<variable> "<help_text>" [value])
>
> 不设置value时默认是OFF。

接下来，我们创建一个EXTRA_LIBS和EXTRA_INCLUDE列表变量使其更加具有拓展性。在if语块中，判断USE_MYMATH的值，若为打开状态，则将其添加进入EXTRA_LIBS和EXTRA_INCLUDES中

```cmake
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()
```

使用target_link_libraries和target_includes将EXTRA_LIBS和EXTRA_INCLUDES添加进目标程序内。

```cmake
target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES}
                           )
```

在头文件内添加判断语句

```cpp
#ifdef USE_MYMATH
#  include "MathFunctions.h"
#endif
```

在源代码中添加判断语句

```cpp
#ifdef USE_MYMATH
  const double outputValue = mysqrt(inputValue);
#else
  const double outputValue = sqrt(inputValue);
#endif
```

在配置文件内添加cmakedefine来给USE_MYMATH赋值。、

```cpp
#cmakedefine USE_MYMATH
```

## Step3 为链接库添加使用要求

### Exe1: 为链接库添加使用要求

目标参数的要求允许我们队库或可执行文件和include行进行更好的控制，同时也可以对CMake中目标的传递属性进行更多的控制，利用使用要求的主要命令有：

- [`target_compile_definitions()`](https://cmake.org/cmake/help/latest/command/target_compile_definitions.html#command:target_compile_definitions)
- [`target_compile_options()`](https://cmake.org/cmake/help/latest/command/target_compile_options.html#command:target_compile_options)
- [`target_include_directories()`](https://cmake.org/cmake/help/latest/command/target_include_directories.html#command:target_include_directories)
- [`target_link_directories()`](https://cmake.org/cmake/help/latest/command/target_link_directories.html#command:target_link_directories)
- [`target_link_options()`](https://cmake.org/cmake/help/latest/command/target_link_options.html#command:target_link_options)
- [`target_precompile_headers()`](https://cmake.org/cmake/help/latest/command/target_precompile_headers.html#command:target_precompile_headers)
- [`target_sources()`](https://cmake.org/cmake/help/latest/command/target_sources.html#command:target_sources)

在本练习中，我们将从添加库中重构代码，使用现代CMake方法，将从我们的库中定义自己的使用要求，以便根据需要将他们传递给其他目标。在这种情况下，MathFunction将自己指定任何需要的include目录，然后，目标Tutorial只需要链接到MathFunction而不必担心任何额外的include目录。

#### Solution:

我们希望什么，任何链接到MathFunction的人都需要包含当前源目录，而MathFunction本身则不包含，这可以通过`INTERFACE`使用需求来表示，INTERFACE是指消费者需要但生产者不需要的东西

在MathFunctions/CMakeLists.txt的借位，使用`target_include_directories`和`INTERFACE`关键字，

```cmake
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
```

接下来，移除顶层CMake中的EXTRA_INCLUDES和target_include_directories。

```cmake
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
endif()
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```



## Step4: 添加生成表达式 generator expressions

### Exe1: 通过Interface Libraries设置C++标准

generator expressions 在构建系统生成期间进行求值，以生成特定于每个构建配置的信息。

generator expressions在许多目标属性的上下文中被允许使用，比如LINK_LIBRARIES, INCLUDE_DIRECTORIES, COMPILE_DEFINITIONS或其他，当使用命令来生成这些属性时也被允许使用，比如target_link_libraries(), target_include_directories, target_compile_definitions() and others。

 generator expressions也被用于启用条件链接，编译时进行条件定义，条件包含目录等。这些条件可能基于构建配置，目标属性，平台信息或其他可查询的信息。

有不同类型的生成器表达式，包括逻辑不表达式、信息表达式或输出表达式。

逻辑表达式由于生成条件输出，这些基础表达式是01表达式，A `$<0:..>`结果对应空字符串，`<1:...>`结果对于内容，可以嵌套。

#### Helpful Resources

- [`add_library()`](https://cmake.org/cmake/help/latest/command/add_library.html#command:add_library)
- [`target_compile_features()`](https://cmake.org/cmake/help/latest/command/target_compile_features.html#command:target_compile_features)
- [`target_link_libraries()`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html#command:target_link_libraries)

#### Solution

首先我们对上一步中的代码进行更新使用interface libraries来设置 C++ 依赖

一开始时，移除两个设置CMAKE_CXX_STANDARD和CMAKE_CXX_STANDARD_REQUIRED变量的语句。

接下来，我们需要创建一个接口库，`tutorial_compiler_flags`，之后使用target_compile_features()来添加编译选项cxx_std_11。

```cmake
add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)
```

最终，由于我们的interface library被设置，我们需要链接可执行文件Target和MathFunctions库到我们新tutorial_compiler_flags库，代码如下图所示

CMakeLists.txt:

```cmake
target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS} tutorial_compiler_flags)
```

MathFunctions/CMakeLists.txt:

```cmake
target_link_libraries(MathFunctions tutorial_compiler_flags)
```

这样我们的代码都可以依赖 C++ 11来构建，注意到通过这种方法，我们代码的标准只有一个接口源。

### Exp2: 通过Generator Expressions添加编译器警告标志

Generator Expressions 的一个常见用法是有条件地添加编译器标志，比如语言级别或警告的标志，一种很好的模式是将这些信息和允许传播该信息的INTERFACE目标相关联。

#### Solution:

首先更新所需的CMake最低版本位3.15:

```cmake
cmake_minimum_required(VERSION 3.15)
```

接下来，我们确定我们的系统当前正在使用哪个编译器进行编译，因为编译器不同警告标志也会不同。这是通过使用`COMPILE_LANG_AND_ID`生成器表达式来完成的，我们在变量`gcc_like_cxx`和`msvc_cxx`变量中设置结果

```cmake
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
```

之后，为我们的项目添加所需的编译器警告标志，使用我们的变量gcc_like_cxx和msvc_cxx。我们可以使用另一个生成器表达式，仅当变量位ture时才使用相应的标志，我们使用target_compile_options()将这些标志应用于接口库。

```cmake
target_compile_options(tutorial_compiler_flags INTERFACE
  "$<${gcc_like_cxx}:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
  "$<${msvc_cxx}:-W3>"
)
```

最后，我们只希望在构建期间使用这些警告标志，我们已安装项目的使用者不应继承我们的警告标志。为此，我们使用BUILD_INTERFACE条件将标志包装在生成器表达式中。

```cmake
target_compile_options(tutorial_compiler_flags INTERFACE
  "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
  "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
)
```

## Step5: 安装与测试

### Exe1：安装规则

通常，仅仅构建一个可执行文件是不够的，它还应该是可安装的。使用CMake，我们可以使用`install()`命令指定安装规则。在CMake中支持构建的本地安装与指定安装位置以及要安装的目标和文件一样简单。

#### Solution：

项目的安装规则

- 对于MathFunctions库，我们希望分别安装库和头文件到lib和include directories。
- 对于可执行文件Tutroial可执行程序，我们希望将可执行文件和配置文件的头文件分别安装到bin和include目录中。

因此在MathFunctions目录下的CMakeLists.txt中指定库和头文件安装目录。

```cmake
set(installable_libs MathFunctions tutorial_compiler_flags)
install(TARGETS ${installable_libs} DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

接下来在顶级CMakeLists.txt目录下指定可执行文件和配置文件的头文件安装目录。

```cmake
install(TARGETS Tutorial DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include
  )
```


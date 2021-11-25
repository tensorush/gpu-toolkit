# :man_technologist: :toolbox: **CUDA Hacker's Toolkit**

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](https://github.com/tensorush/CUDA-Hackers-Toolkit/pulls)
[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![MIT License][license-shield]][license-url]

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/tensorush/CUDA-Hackers-Toolkit)
[![forthebadge](https://forthebadge.com/images/badges/works-on-my-machine.svg)](https://forthebadge.com)

<p align="center">
    <img src="https://bleuje.github.io/gifset/2021/gifs/2021_8_cyclespropagation_v2.gif">
</p>

<h4 align="center"> 
    <p><a href="https://twitter.com/etiennejcb/">Animation by Étienne Jacob</a></p>
</h4>

## :axe: Essential algorithm kernels for GPU hacking with CUDA C++ that I've been researching, reimplementing and refining for quick reference.

> ### _Any contributions, corrections or optimizations are very welcome!_ :hugs:

### :warning: Purpose of this repo is to showcase minimal CUDA C++ kernel implementations of popular algorithms. For this reason most implementations aren't optimized and contain only GPU kernels. If you seek optimized solutions you should refer to [NVIDIA CUDA-X Libraries](https://developer.nvidia.com/gpu-accelerated-libraries). In case you're curious about optimizing custom kernels, have a look at the following CUDA C++ programs:

- #### Multi-stream execution - [Matrix addition program](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Mathematical-Algorithms/Linear-Algebra/matrix_addition.cu)

- #### Shared memory allocation - [Square matrix multiplication program](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Mathematical-Algorithms/Linear-Algebra/square_matrix_multiplication.cu)

- #### Texture memory allocation - [Three-digit Armstrong numbers program](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Mathematical-Algorithms/Number-Theory/three-digit_armstrong-numbers.cu)

## :axe: [Algorithms](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms)

- ### :bus: [Array Algorithms](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Array-Algorithms)

- ### :framed_picture: [Image Processing Algorithms](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Image-Processing-Algorithms)

- ### :dragon: [Computer Graphics Algorithms](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Computer-Graphics-Algorithms)

- ### :scroll: [Mathematical Algorithms](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Mathematical-Algorithms)

  - ### :mechanical_arm: [Linear Algebra](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Mathematical-Algorithms/Linear-Algebra)

  - ### :abacus: [Number Theory](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Mathematical-Algorithms/Number-Theory)

- ### :bar_chart: [Sorting Algorithms](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/Sorting-Algorithms)

- ### :dna: [String Algorithms](https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/Algorithms/String-Algorithms)

## :man_teacher: Learning Resources

### :film_projector: [CUDA Streams](https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf)

### :tv: [CUDA Tutorials by Creel](https://www.youtube.com/playlist?list=PLKK11Ligqititws0ZOoGk3SW-TZCar4dK)

### :man_technologist: [CUDA Samples by NVIDIA](https://github.com/NVIDIA/cuda-samples)

### :man_technologist: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#abstract)

### :tv: [Intro to CUDA by Josh Holloway](https://www.youtube.com/playlist?list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe)

### :man_technologist: [CUDA Training Series by NVIDIA](https://github.com/olcf/cuda-training-series)

### :man_technologist: [CUDA Examples by Sylvain Jubertie](https://github.com/sjubertie/teaching-CUDA/tree/master/examples)

### :film_projector: [CUDA C/C++ Basics Slides by NVIDIA](https://www.olcf.ornl.gov/wp-content/uploads/2013/02/Intro_to_CUDA_C-TS.pdf)

### :tv: [CUDA Crash Course by CoffeeBeforeArch](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)

### :film_projector: [GPU Memory Slides by Robert Dalrymple](https://www.ce.jhu.edu/dalrymple/classes/602/Class13.pdf)

### :thought_balloon: [CUDA Developer Blogposts by Mark Harris](https://developer.nvidia.com/blog/author/mharris/)

### :film_projector: [Parallel Prefix Sum with CUDA by Mark Harris](https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf)

### :thought_balloon: [CUDA Basics Blogposts by Sebastian Eberhardt](https://gpgpu.io/category/cuda-basics/)

### :tv: [Parallel and Distributed Computing by Tom Nurkkala](https://www.youtube.com/playlist?list=PLG3vBTUJlY2HdwYsdFCdXQraInoc3j9DU)

### :film_projector: [Advanced CUDA: Memory Optimization Slides by NVIDIA](https://on-demand.gputechconf.com/gtc-express/2011/presentations/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf)

### :film_projector: [Optimizing Parallel Reduction in CUDA Slides by Mark Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

### :film_projector: [Advanced CUDA: Performance Optimization Slides by NVIDIA](https://www.nvidia.com/content/cudazone/download/Advanced_CUDA_Training_NVISION08.pdf)

### :film_projector: [Better Performance at Lower Occupancy Slides by Vasily Volkov](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf)

### :film_projector: [CUTLASS: CUDA Template Library Slides by Andrew Kerr and others](https://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf)

### :film_projector: [GPU Performance Analysis and Optimization by Paulius Micikevicius](https://on-demand.gputechconf.com/gtc/2012/presentations/S0514-GTC2012-GPU-Performance-Analysis.pdf)

### :film_projector: [General-purpose GPU Computing with CUDA Course by Will Landau](https://github.com/wlandau/gpu/tree/main/lectures)

### :man_technologist: [Learn CUDA Programming by Jaegeun Han and Bharatkumar Sharma](https://github.com/PacktPublishing/Learn-CUDA-Programming)

### :man_teacher: [CUDA and Applications to Task-based Programming Eurographics 2021 Tutorial](https://cuda-tutorial.github.io/)

### :tv: [Introduction to GPU Programming with CUDA and Thrust Talk by Richard Thomson](https://www.youtube.com/watch?v=tbb835UFRQ4&t=2229s)

### :thought_balloon: [CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics by Andy Adinets](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

<!-- MARKDOWN LINKS -->

[contributors-shield]: https://img.shields.io/github/contributors/tensorush/CUDA-Hackers-Toolkit.svg?style=for-the-badge
[contributors-url]: https://github.com/tensorush/CUDA-Hackers-Toolkit/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/tensorush/CUDA-Hackers-Toolkit.svg?style=for-the-badge
[issues-url]: https://github.com/tensorush/CUDA-Hackers-Toolkit/issues
[stars-shield]: https://img.shields.io/github/stars/tensorush/CUDA-Hackers-Toolkit.svg?style=for-the-badge
[stars-url]: https://github.com/tensorush/CUDA-Hackers-Toolkit/stargazers
[forks-shield]: https://img.shields.io/github/forks/tensorush/CUDA-Hackers-Toolkit.svg?style=for-the-badge
[forks-url]: https://github.com/tensorush/CUDA-Hackers-Toolkit/network/members
[license-shield]: https://img.shields.io/github/license/tensorush/CUDA-Hackers-Toolkit.svg?style=for-the-badge
[license-url]: https://github.com/tensorush/CUDA-Hackers-Toolkit/blob/master/LICENSE.md

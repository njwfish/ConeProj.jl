# ConeProj

[![Build Status](https://github.com/njwfish/ConeProj.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/njwfish/ConeProj.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package implements a cone projection algorithm outlined [here](https://www.jstatsoft.org/article/view/v061i12) in pure Julia. 

The algorithm is designed for efficiently solving non-negative least squares problems. In this Julia version we speed up the package considerably leveraging Q-less updatable QR factorizations to remove the need to solve the least squares problem at each iteration. We additionally develop a novel extension of the algorithm to handle the equality constrained non-negative least squares problem.

The primary methods are `nnls` and `ecnnls` which solve the non-negative least squares problem and equality constrained non-negative least squares problem respectively. 
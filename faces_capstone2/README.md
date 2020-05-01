# Capstone 2: Hybrid Face Generator

This project is the second capstone for the Springboard Data Science course. The goal was to see if a variational autoencoder is able to learn features of cat and human faces in order to create combined faces. This project was inspired by and modeled after CodeParade's YouTube video<sup name="a1">[1](#f1)</sup> and source code<sup name="a2">[2](#f2)</sup>, where they generate unique faces based on images taken from a high school yearbook.

## Steps

This project consisted of five stages:

1. Data Acquisition - cat dataset was downloaded from Kaggle and face dataset was requested from principal investigators who previously prepared the data for a paper 
2. Data Cleaning - cropping cat images based on provided annotations, writing shell script to move images, and resizing images
3. Model Building and Training - first a large scale model, then a reduced model; periodically backing up project files
4. Creating Application - making an interface to allow user to adjust sliders that controlled the latent space vectors
5. Writing Final Report - visualizing model performance and creating final write-up of project

[further explanation goes here]

<sup><b name="f1">1</b></sup> https://www.youtube.com/watch?v=4VAkrUNLKSo&t=1s[↩](#a1)

<sup><b name="f2">2</b></sup> https://github.com/HackerPoet/FaceEditor[↩](#a2)

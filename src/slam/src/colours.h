#pragma once

#include <vector>

struct Colour {
    float r;
    float g;
    float b;
};

std::vector<Colour> cone_colours = {
    {0, 0, 1},        // blue
    {0.9, 0.9, 0.9},  // unknown
    {1, 0.64, 0},     // big orange
    {1, 0.64, 0},     // small orange
    {1, 1, 0},        // yellow
};

std::vector<Colour> colours = {
    {0.61, 0.77, 0.9},  {0.19, 0.0, 0.02},  {0.02, 0.39, 0.05}, {1.0, 0.98, 0.04},  {0.98, 0.33, 0.08},
    {0.88, 0.08, 0.75}, {0.0, 0.35, 0.5},   {0.04, 0.77, 0.51}, {1.0, 0.72, 0.78},  {0.62, 0.51, 0.09},
    {0.0, 0.1, 0.06},   {0.52, 0.49, 0.51}, {0.35, 0.0, 0.55},  {0.72, 0.02, 0.22}, {0.44, 0.23, 0.0},
    {0.97, 0.95, 0.87}, {0.07, 0.55, 0.54}, {0.29, 1.0, 0.98},  {0.99, 0.69, 0.39}, {0.47, 0.43, 0.9},
    {0.0, 0.05, 0.17},  {0.33, 0.29, 0.37}, {0.98, 0.33, 0.46}, {0.38, 0.99, 0.01}, {0.36, 0.59, 0.03},
    {0.87, 0.6, 0.99},  {0.6, 0.63, 0.53},  {0.31, 0.35, 0.31}, {0.14, 0.54, 0.82}, {0.36, 0.33, 0.0},
    {0.62, 0.4, 0.32},  {0.74, 1.0, 0.78},  {0.58, 0.17, 0.44}, {0.17, 0.11, 0.02}, {0.71, 0.69, 0.77},
    {0.83, 0.78, 0.48}, {0.68, 0.48, 0.63}, {0.76, 0.64, 0.58}, {0.01, 0.2, 0.99},  {0.42, 0.23, 0.21},
    {0.73, 0.41, 0.0},  {0.09, 0.56, 0.36}, {0.09, 0.75, 0.82}, {0.78, 0.13, 0.0},  {0.0, 0.26, 0.28},
    {0.14, 0.22, 0.04}, {0.26, 0.03, 0.23}, {0.51, 0.47, 0.36}, {0.01, 0.19, 0.53}, {0.72, 0.85, 0.82},
    {0.1, 0.41, 0.34},  {0.55, 0.25, 0.73}, {0.93, 0.93, 1.0},  {0.17, 0.18, 0.2},  {0.58, 0.78, 0.38},
    {0.97, 0.56, 0.49}, {0.54, 0.37, 0.42}, {0.47, 0.56, 0.58}, {0.98, 0.42, 0.72}, {0.34, 0.38, 0.58},
    {0.86, 0.08, 0.45}, {0.52, 0.54, 0.68}, {0.53, 0.05, 0.02}, {0.98, 0.76, 0.02}, {0.43, 0.67, 0.61},
    {0.95, 0.8, 1.0},   {0.39, 0.33, 0.25}, {0.46, 0.0, 0.21},  {0.39, 0.48, 0.25}, {0.29, 0.43, 0.46},
    {0.89, 0.97, 0.58}, {0.98, 0.84, 0.8},  {0.53, 0.38, 0.16}, {0.63, 0.65, 0.07}, {0.0, 0.98, 0.57},
    {0.99, 0.06, 0.19}, {0.75, 0.52, 0.52}, {0.78, 0.38, 0.98}, {0.07, 0.0, 0.02},  {0.83, 0.54, 0.35},
    {0.02, 0.68, 0.91}, {0.76, 0.76, 0.75}, {0.62, 0.6, 0.97},  {0.07, 0.4, 0.85},  {0.82, 0.56, 0.07},
    {0.72, 0.85, 0.01}, {0.51, 0.39, 0.57}, {0.37, 0.48, 0.42}, {0.7, 0.6, 0.41},   {0.11, 0.0, 0.32},
    {0.55, 0.91, 0.99}, {0.46, 0.88, 0.76}, {0.73, 0.81, 0.65}, {0.07, 0.73, 0.04}, {0.27, 0.17, 0.21},
    {0.4, 0.25, 0.49},  {0.29, 0.09, 0.01}, {0.96, 0.82, 0.66}, {0.01, 0.26, 0.17}, {0.45, 0.64, 0.43},
    {0.07, 0.56, 0.67}, {0.28, 0.33, 0.37}, {0.73, 0.36, 0.41}, {0.63, 0.3, 0.07},  {0.77, 0.78, 0.98},
    {0.22, 0.16, 0.33}, {0.25, 0.21, 0.06}, {0.83, 0.64, 0.78}, {0.44, 0.62, 0.98}, {0.05, 0.52, 0.1},
    {0.3, 0.36, 0.2},   {0.62, 0.7, 0.72},  {0.69, 0.31, 0.56}, {0.45, 0.44, 0.01}, {0.62, 0.51, 0.43},
    {0.82, 0.42, 0.36}, {0.55, 0.58, 0.29}, {0.98, 0.52, 0.0},  {0.0, 0.16, 0.21},  {0.84, 0.95, 1.0},
    {0.99, 0.72, 0.6},  {0.11, 0.03, 0.13}, {0.42, 0.37, 0.38}, {0.98, 0.54, 0.62}, {0.61, 0.45, 0.76},
    {0.65, 0.57, 0.62}, {0.17, 0.22, 0.16}, {0.84, 0.78, 0.04}, {0.62, 0.6, 0.57},  {0.94, 0.98, 0.82},
    {0.99, 0.89, 0.95}, {0.57, 0.23, 0.32}, {0.32, 0.25, 0.65}, {0.74, 0.08, 0.99}, {0.43, 0.44, 0.42},
    {0.0, 0.03, 0.77},  {0.78, 0.65, 0.18}, {0.0, 0.05, 0.08},  {0.56, 0.27, 0.19}, {0.38, 0.0, 0.07},
    {0.11, 0.11, 0.03}, {0.41, 0.22, 0.33}, {0.37, 0.49, 0.6},  {0.42, 0.43, 0.51}, {0.82, 0.69, 0.7},
    {0.29, 0.23, 0.21}, {0.67, 0.58, 0.81}, {0.77, 0.73, 0.61}, {0.04, 0.77, 0.72}, {0.41, 0.65, 0.72},
    {0.22, 0.28, 0.41}, {0.97, 0.41, 0.93}, {0.91, 0.03, 0.31}, {0.75, 0.28, 0.25}, {0.76, 0.39, 0.2},
    {0.44, 0.01, 0.4},  {0.54, 0.48, 0.58}, {0.32, 0.21, 0.11}, {0.71, 0.01, 0.64}, {0.82, 0.44, 0.56},
    {0.63, 0.94, 0.53}, {0.48, 0.25, 0.99}, {0.05, 0.65, 0.31}, {0.0, 0.45, 0.6},   {0.03, 0.66, 0.51},
    {0.45, 0.0, 0.8},   {0.66, 0.69, 0.45}, {0.31, 0.39, 0.0},  {0.67, 0.49, 0.25}, {0.33, 0.5, 0.96},
    {0.07, 0.3, 0.67},  {0.99, 0.93, 0.53}, {0.02, 0.38, 0.39}, {1.0, 0.07, 0.63},  {0.76, 0.39, 0.73},
    {0.58, 0.62, 0.68}, {0.04, 0.8, 0.98},  {0.15, 0.45, 0.26}, {0.11, 0.87, 0.29}, {0.51, 0.41, 0.35},
    {0.59, 0.46, 0.47}, {0.73, 0.99, 0.91}, {0.49, 0.52, 0.46}, {0.55, 0.81, 0.58}, {0.45, 0.4, 0.22},
    {1.0, 0.66, 0.92},  {0.92, 1.0, 0.94},  {0.42, 0.57, 0.47}, {0.76, 1.0, 0.29},  {0.19, 0.25, 0.25},
    {0.12, 0.65, 0.65}, {0.01, 0.14, 0.01}, {0.02, 0.16, 0.28}, {0.02, 0.29, 0.09}, {0.96, 0.78, 0.45},
    {0.01, 1.0, 0.78},  {0.62, 0.73, 0.66}, {0.47, 0.33, 0.32}, {0.51, 0.33, 0.21}, {0.34, 0.36, 0.8},
    {0.5, 0.84, 0.82},  {0.48, 0.84, 0.03}, {0.41, 0.44, 0.33}, {0.53, 0.03, 0.6},  {0.4, 0.29, 0.1},
    {0.14, 0.13, 0.21}, {0.49, 0.69, 0.05}, {0.75, 0.78, 0.84}, {0.84, 0.66, 0.49}, {0.26, 0.25, 0.19},
    {0.19, 0.1, 0.09},  {0.99, 0.7, 0.67},  {0.84, 0.53, 0.79}, {0.48, 0.37, 0.69}, {0.2, 0.33, 0.29},
    {0.94, 0.89, 0.69}, {0.52, 0.62, 0.59}, {0.17, 0.52, 0.44}, {0.55, 0.16, 0.18}, {0.88, 0.42, 0.03},
    {0.29, 0.0, 0.15},  {0.01, 0.06, 0.51}, {0.07, 0.27, 0.35}, {0.97, 0.03, 0.98}, {0.78, 0.52, 0.44},
    {0.5, 0.73, 0.74},  {0.99, 0.5, 0.29},  {0.55, 0.29, 0.57}, {0.42, 0.19, 0.1},  {0.53, 0.31, 0.45},
    {0.6, 0.31, 0.31},  {0.62, 0.66, 0.83}, {0.53, 0.48, 0.25}, {0.81, 0.84, 0.77}, {0.11, 0.64, 1.0},
    {0.85, 0.77, 0.71}, {1.0, 0.67, 0.0},   {0.31, 0.48, 0.0},  {0.65, 0.82, 0.86}, {0.33, 0.52, 0.55},
    {0.35, 0.56, 0.29}, {0.98, 0.93, 0.93}, {0.99, 0.58, 0.76}, {0.84, 0.8, 0.83},  {0.24, 0.29, 0.01},
    {0.78, 0.69, 0.89}, {0.48, 0.55, 0.38}, {0.6, 0.35, 0.89},  {0.54, 0.42, 0.02}, {0.69, 0.07, 0.11},
    {0.25, 0.18, 0.49}, {0.52, 0.53, 0.0},  {0.83, 0.6, 0.65},  {0.71, 0.52, 0.94}, {0.36, 0.28, 0.3},
    {0.02, 0.47, 0.51}, {0.75, 0.98, 0.99}, {0.45, 0.38, 0.46}, {0.55, 0.19, 0.0},  {0.42, 0.58, 0.7},
    {0.64, 0.42, 0.25}, {0.67, 0.4, 0.51},  {0.31, 0.3, 0.31},  {0.35, 0.34, 0.24}, {0.91, 0.19, 0.02},
    {0.2, 0.29, 0.18},  {0.99, 0.45, 0.45}, {0.73, 0.77, 0.34}, {0.33, 0.16, 0.36}, {0.71, 0.02, 0.39},
    {0.38, 0.43, 0.47}, {0.86, 0.89, 0.89}, {0.81, 0.5, 0.16},  {0.04, 0.89, 0.94}, {0.31, 0.12, 0.14},
    {0.99, 0.37, 0.27}, {0.29, 0.41, 0.31}, {0.77, 0.87, 0.99}, {0.36, 0.76, 0.38}, {0.01, 0.18, 0.15},
    {0.47, 0.46, 0.72}, {0.99, 0.62, 0.4},  {0.69, 0.29, 0.72}, {0.6, 0.56, 0.45},  {0.75, 0.22, 0.35},
    {0.17, 0.13, 0.15}, {0.33, 0.5, 0.35},  {0.08, 0.11, 0.33}, {0.4, 0.75, 0.61},  {0.27, 0.41, 0.54},
    {0.87, 0.76, 0.85}, {0.09, 0.38, 0.46}, {0.76, 0.89, 0.61}, {0.64, 0.59, 0.71}, {0.18, 0.16, 0.13},
    {0.67, 0.86, 0.75}, {0.71, 0.65, 0.66}, {0.63, 0.42, 0.03}, {0.66, 0.6, 0.29},  {0.04, 0.02, 0.09},
    {0.69, 0.31, 0.18}, {0.38, 0.33, 0.49}, {0.83, 0.65, 0.34}, {0.51, 0.65, 0.32}, {0.29, 0.0, 0.36},
    {0.24, 0.25, 0.31}, {0.43, 0.4, 0.34},  {0.49, 0.55, 0.84}, {0.07, 0.46, 0.72}, {0.84, 0.62, 0.57},
    {0.14, 0.03, 0.21}, {0.4, 0.09, 0.29},  {0.48, 0.51, 0.57}, {1.0, 0.06, 0.48},  {0.69, 0.71, 0.66},
    {0.38, 0.58, 0.57}, {0.82, 0.33, 0.57}, {0.59, 0.71, 0.54}, {0.59, 0.58, 0.6},  {0.01, 0.37, 0.22},
    {0.33, 0.88, 0.62}, {0.87, 0.84, 0.98}, {0.01, 0.26, 0.42}, {0.32, 0.35, 0.45}, {0.02, 0.6, 0.05},
    {0.24, 0.45, 0.42}, {0.67, 0.56, 0.53}, {0.82, 0.05, 0.57}, {0.73, 0.56, 0.43}, {0.4, 0.74, 0.99},
    {0.75, 0.67, 0.99}, {0.03, 0.2, 0.74},  {0.2, 0.07, 0.14},  {0.54, 0.67, 0.76}, {0.05, 0.04, 0.01},
    {0.25, 0.27, 0.13}, {0.42, 0.18, 0.24}, {0.18, 0.6, 0.54},  {0.27, 0.41, 0.99}, {0.99, 0.9, 0.82},
    {1.0, 0.88, 0.03},  {0.6, 0.0, 0.24},   {0.67, 0.51, 0.56}, {0.86, 0.87, 0.35}, {0.72, 0.56, 0.24},
    {0.12, 0.16, 0.15}, {0.61, 0.01, 0.9},  {0.51, 0.48, 0.44}, {0.53, 0.55, 0.54}, {0.56, 0.45, 0.31},
    {0.67, 0.29, 0.44}, {0.22, 0.14, 0.23}, {0.22, 0.33, 0.35}, {0.95, 0.28, 0.78}, {0.62, 0.71, 1.0},
    {0.84, 0.44, 0.47}, {0.87, 0.31, 0.35}, {0.22, 0.97, 0.87}, {0.31, 0.21, 0.0},  {0.11, 0.14, 0.0},
    {0.87, 0.01, 0.14}, {0.0, 0.64, 0.73},  {0.58, 0.34, 0.01}, {0.98, 0.36, 0.58}, {0.67, 0.46, 0.42},
    {0.72, 0.88, 0.4},  {0.42, 0.5, 0.49},  {0.3, 0.18, 0.15},  {0.45, 0.75, 0.84}, {0.84, 0.74, 0.54},
    {0.38, 0.27, 0.22}, {0.32, 0.41, 0.38}, {0.44, 0.43, 0.59}, {0.51, 0.6, 0.09},  {0.13, 0.0, 0.04},
    {0.26, 0.42, 0.18}, {0.47, 0.29, 0.33}, {0.6, 0.48, 0.67},  {0.56, 0.0, 0.32},  {0.02, 0.32, 0.98},
    {0.71, 0.47, 0.34}, {0.63, 0.4, 0.62},  {0.83, 0.97, 0.85}, {0.28, 0.25, 0.44}, {0.87, 0.73, 0.69},
    {0.65, 0.66, 0.67}, {0.55, 0.42, 0.51}, {0.25, 0.22, 0.25}, {0.44, 0.53, 0.17}, {0.85, 0.45, 0.3},
    {0.08, 0.12, 0.17}, {0.36, 0.37, 0.37}, {0.71, 0.49, 0.01}, {0.96, 0.8, 0.82},  {0.89, 0.62, 0.49},
    {0.87, 0.6, 0.33},  {0.69, 0.63, 0.55}, {0.17, 0.33, 0.03}, {0.93, 0.99, 0.39}, {0.62, 0.45, 0.99},
    {0.16, 0.2, 0.32},  {0.41, 0.29, 0.42}, {0.79, 0.28, 0.0},  {0.93, 0.82, 0.37}, {0.51, 0.44, 0.43},
    {0.88, 0.84, 0.73}, {0.36, 0.43, 0.71}, {0.4, 0.18, 0.6},   {0.05, 0.59, 0.79}, {0.76, 0.79, 0.54},
    {0.46, 0.35, 0.01}, {0.87, 0.65, 0.1},  {0.8, 0.44, 0.66},  {0.73, 0.79, 0.78}, {0.96, 0.74, 0.89},
    {0.63, 0.39, 0.38}, {0.0, 0.82, 0.67},  {0.53, 0.78, 0.7},  {0.91, 0.7, 0.98},  {0.85, 0.33, 0.47},
    {0.39, 0.23, 0.84}, {0.82, 0.54, 0.68}, {0.07, 0.99, 0.37}, {0.7, 0.89, 0.99},  {0.79, 0.47, 0.86},
    {0.76, 0.65, 0.73}, {0.57, 0.53, 0.8},  {0.63, 0.61, 0.42}, {0.56, 1.0, 0.84},  {0.42, 0.12, 0.09},
    {0.87, 0.31, 0.23}, {0.06, 0.87, 0.84}, {0.6, 0.52, 0.34},  {0.38, 0.4, 0.18},  {0.49, 0.2, 0.49},
    {0.87, 0.53, 0.51}, {0.35, 0.67, 0.26}, {0.51, 0.99, 0.72}, {0.99, 0.54, 0.91}, {0.56, 0.62, 0.44},
    {0.71, 0.57, 0.68}, {0.72, 0.07, 0.8},  {0.74, 0.7, 0.31},  {0.8, 0.29, 0.85},  {0.17, 0.14, 0.02},
    {0.67, 0.58, 0.0},  {0.36, 0.31, 0.59}, {0.25, 0.2, 0.13},  {0.98, 0.98, 0.71}, {0.22, 0.56, 0.99},
    {0.44, 0.87, 0.5},  {0.58, 0.52, 0.5},  {0.52, 0.64, 0.52}, {0.31, 0.6, 0.44},  {0.47, 0.48, 0.33},
    {0.48, 0.38, 0.26}, {0.51, 0.84, 1.0},  {0.61, 0.77, 0.16}, {0.04, 0.02, 0.22}, {0.24, 0.13, 0.02},
    {0.29, 0.49, 0.57}, {0.32, 0.22, 0.33}, {0.0, 0.37, 0.66},  {0.94, 0.78, 0.68}, {0.67, 0.72, 0.6},
    {0.98, 0.75, 0.56}, {0.31, 0.13, 0.22}, {0.75, 0.67, 0.42}, {0.17, 0.24, 0.28}, {0.05, 0.71, 0.85},
    {0.54, 0.34, 0.28}, {0.29, 0.69, 0.45}, {0.02, 0.48, 0.91}, {0.95, 0.58, 0.04}, {0.33, 0.27, 0.16},
    {0.27, 0.15, 0.64}, {0.45, 0.32, 0.79}, {0.25, 0.26, 0.53}, {0.55, 0.4, 0.37},  {0.71, 0.5, 0.75},
    {0.61, 0.65, 0.3},  {0.37, 0.32, 0.3},  {0.8, 0.61, 0.86},  {0.73, 0.47, 0.26}, {0.11, 0.25, 0.22},
    {0.24, 0.24, 0.23}, {0.16, 0.69, 0.61}, {0.01, 0.57, 0.25}, {0.44, 0.11, 0.17}, {0.21, 0.34, 0.49},
    {0.25, 0.0, 0.92},  {0.24, 0.58, 0.62}, {0.27, 0.02, 0.0},  {0.54, 0.94, 0.95}, {0.43, 0.27, 0.16},
    {0.75, 0.69, 0.66}, {0.63, 0.11, 0.01}, {0.51, 0.51, 1.0},  {0.65, 0.22, 0.22}, {0.86, 0.87, 0.54},
    {0.01, 0.51, 0.7},  {0.53, 0.52, 0.59}, {0.2, 0.35, 0.18},  {0.96, 0.99, 0.98}, {0.0, 0.1, 0.11},
    {0.67, 0.44, 0.48}, {0.71, 0.74, 0.01}, {0.01, 0.48, 0.35}, {0.48, 0.31, 0.03}, {0.58, 0.47, 0.22},
    {0.51, 0.45, 0.49}, {0.01, 0.33, 0.26}, {0.44, 0.49, 0.39}, {0.76, 0.6, 0.6},   {0.32, 0.52, 0.48},
    {0.57, 0.35, 0.67}, {0.47, 0.81, 0.85}, {0.32, 0.39, 0.41}, {0.88, 0.84, 0.82}, {0.99, 0.87, 0.59},
    {0.33, 0.33, 0.14}, {0.59, 0.9, 0.71},  {0.52, 0.73, 0.45}, {0.37, 0.13, 0.45}, {0.74, 0.37, 0.28},
    {0.61, 0.93, 0.33}, {0.1, 0.21, 0.12},  {0.19, 0.28, 0.8},  {0.44, 0.34, 0.37}, {0.41, 0.65, 0.82},
    {0.22, 0.1, 0.38},  {0.91, 0.62, 0.63}, {0.11, 0.06, 0.01}, {0.11, 0.09, 0.21}, {0.82, 0.05, 0.22},
    {0.46, 0.33, 0.59}, {0.45, 0.01, 1.0},  {0.27, 0.5, 0.24},  {0.81, 0.82, 0.66}, {0.23, 0.15, 0.0},
    {0.41, 0.35, 0.99}, {0.64, 0.7, 0.78},  {0.33, 0.26, 0.01}, {0.6, 0.63, 0.59},  {0.99, 0.32, 0.33},
    {0.61, 0.0, 0.52},  {0.25, 0.22, 0.34}, {0.5, 0.63, 0.65},  {0.43, 0.48, 0.6},  {0.38, 0.37, 0.42},
    {0.53, 0.94, 0.89}, {0.35, 0.17, 0.0},  {0.49, 0.24, 0.26}, {0.93, 0.51, 0.23}, {0.2, 0.2, 0.11},
    {0.26, 0.28, 0.22}, {0.25, 0.46, 0.37}, {0.32, 0.31, 0.28}, {0.72, 0.35, 0.03}, {0.71, 0.0, 0.5},
    {0.36, 0.55, 0.63}, {0.99, 0.81, 0.9},  {0.8, 1.0, 0.67},   {0.46, 0.35, 0.28}, {0.79, 0.7, 0.59},
    {0.75, 0.84, 0.89}, {0.18, 0.44, 0.0},  {0.84, 0.89, 0.87}, {0.21, 0.16, 0.14}, {0.41, 0.78, 0.24},
    {0.67, 0.22, 0.0},  {0.09, 0.19, 0.2},  {0.28, 0.31, 0.65}, {0.38, 0.72, 0.7},  {0.99, 0.77, 0.71},
    {0.87, 0.73, 0.18}, {1.0, 0.02, 0.29},  {0.45, 0.47, 0.19}, {0.52, 0.44, 0.67}, {0.41, 0.49, 0.53},
    {0.84, 0.72, 0.38}, {0.42, 0.67, 0.53}, {0.51, 0.6, 0.72},  {0.72, 0.71, 0.75}, {0.57, 0.77, 0.63},
    {0.71, 0.03, 0.31}, {0.52, 0.23, 0.37}, {0.82, 0.74, 0.73}, {0.57, 0.51, 0.43}, {0.78, 0.87, 0.78},
    {0.75, 0.37, 0.35}, {0.16, 0.0, 0.13},  {0.26, 0.34, 0.26}, {0.53, 0.27, 0.08}, {0.39, 0.4, 0.35},
    {0.91, 0.47, 0.39}, {0.56, 0.61, 0.62}, {0.6, 0.32, 0.38},  {0.56, 0.56, 0.51}, {0.01, 0.21, 0.03},
    {0.87, 0.68, 0.75}, {0.84, 0.52, 0.58}, {0.21, 0.22, 0.0},  {0.36, 0.0, 0.13},  {0.38, 0.24, 0.28},
    {0.76, 0.58, 0.36}, {0.67, 0.38, 0.8},  {0.99, 0.48, 0.65}, {0.44, 0.42, 0.45}, {0.55, 0.54, 0.36},
    {0.03, 0.06, 0.0},  {0.51, 0.71, 0.95}, {0.71, 0.73, 0.85}, {0.44, 0.53, 0.48}, {0.55, 0.62, 0.89},
    {0.6, 0.44, 0.35},  {0.4, 0.65, 0.67},  {0.18, 0.19, 0.4},  {0.2, 0.07, 0.0},   {1.0, 0.93, 0.8},
    {0.23, 0.37, 0.45}, {0.78, 1.0, 0.52},  {0.63, 0.86, 0.87}, {0.8, 0.29, 0.65},  {0.69, 0.77, 0.89},
    {0.24, 0.37, 0.69}, {0.53, 0.68, 0.65}, {0.02, 0.31, 0.3},  {0.59, 0.32, 0.2},  {0.4, 0.53, 0.73},
    {0.02, 0.53, 0.59}, {0.6, 0.6, 0.77},   {0.63, 0.76, 0.76}, {0.11, 0.22, 0.4},  {0.86, 0.92, 0.03},
    {0.47, 0.59, 0.35}, {0.91, 0.91, 0.78}, {0.65, 0.78, 0.53}, {0.58, 0.5, 0.54},  {0.46, 0.18, 0.38},
    {0.09, 0.08, 0.09}, {0.65, 0.34, 0.28}, {0.0, 0.82, 0.44},  {0.06, 0.33, 0.36}, {0.02, 0.49, 0.46},
    {0.77, 0.28, 0.33}, {0.36, 0.43, 0.53}, {0.67, 0.58, 0.51}, {0.5, 0.23, 0.6},   {0.98, 0.61, 0.28},
    {0.29, 0.54, 0.13}, {0.4, 0.29, 0.36},  {0.59, 0.37, 0.53}, {0.62, 0.05, 0.73}, {0.63, 0.91, 0.63},
    {0.83, 0.86, 0.98}, {0.99, 0.56, 0.56}, {0.68, 0.67, 0.52}, {0.63, 0.23, 0.54}, {0.95, 0.7, 0.31},
    {0.02, 0.41, 0.6},  {0.58, 0.54, 0.26}, {0.78, 0.75, 0.87}, {0.1, 0.15, 0.17},  {0.44, 0.27, 0.67},
    {0.88, 0.93, 0.99}, {0.24, 0.4, 0.34},  {0.8, 0.25, 0.15},  {0.17, 0.1, 0.15},  {0.87, 0.68, 0.58},
    {0.75, 0.69, 0.04}, {0.22, 0.87, 1.0},  {0.01, 0.59, 0.46}, {0.56, 0.45, 0.41}, {0.62, 0.53, 0.65},
    {0.23, 0.11, 0.29}, {0.75, 0.9, 0.72},  {0.76, 0.58, 0.0},  {0.62, 0.21, 0.27}, {0.86, 0.35, 0.04},
    {0.39, 0.34, 0.19}, {0.27, 0.29, 0.29}, {0.99, 0.1, 0.39},  {0.87, 0.9, 0.68},  {0.53, 0.47, 0.0},
    {0.21, 0.0, 0.44},  {0.23, 0.38, 0.38}, {0.47, 0.27, 0.22}, {1.0, 0.63, 0.72},  {0.64, 0.88, 0.82},
    {0.43, 0.39, 0.09}, {0.37, 0.44, 0.45}, {0.73, 0.62, 0.78}, {0.47, 0.48, 0.49}, {0.88, 1.0, 0.99},
    {0.88, 0.43, 0.77}, {0.0, 0.2, 0.29},   {0.97, 0.97, 0.99}, {0.62, 0.62, 0.71}, {0.09, 0.15, 0.09},
    {1.0, 0.24, 0.13},  {0.49, 0.0, 0.09},  {0.51, 0.18, 0.13}, {0.94, 0.85, 0.86}, {0.43, 0.41, 0.77},
    {0.21, 0.28, 0.24}, {0.0, 0.46, 0.14},  {0.46, 0.46, 0.4},  {0.65, 0.51, 0.36}, {0.51, 0.86, 0.37},
    {0.13, 0.45, 0.52}, {0.66, 0.37, 0.2},  {0.32, 0.38, 0.45}, {0.59, 0.59, 0.19}, {0.46, 0.44, 0.43},
    {0.44, 0.38, 0.35}, {0.91, 0.7, 0.71},  {0.71, 0.79, 0.73}, {0.56, 0.47, 0.85}, {0.31, 0.2, 0.43},
    {0.7, 0.22, 0.48},  {0.53, 0.55, 0.44}, {0.19, 0.29, 0.37}, {0.9, 0.71, 0.47},  {0.22, 0.64, 0.78},
    {0.35, 0.38, 0.28}, {0.36, 0.32, 0.36}, {0.8, 0.8, 0.88},   {0.78, 0.59, 0.5},
};

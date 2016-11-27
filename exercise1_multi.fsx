module Exercise1_multi

#load "exercise1.fsx"
#load "packages/FsLab/FsLab.fsx"
#I "packages/MathNet.Numerics.Data.Matlab/lib/net40/"
#I "packages/MathNet.Numerics.Data.Text/lib/net40/"
#r "MathNet.Numerics.Data.Matlab.dll"
#r "MathNet.Numerics.Data.Text.dll"

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Data.Text
open MathNet.Numerics.Statistics;
open XPlot.GoogleCharts
let data = DelimitedReader.Read<double>(
             "/Users/carsten/Projects/courseraMachineLearning/machine-learning-ex1/ex1/ex1data2.txt",
             false, ",", true);

let X' = data.RemoveColumn(2)
let y = data.Column(2)

let m = y.Count

let featureNormalize (m : Matrix<double>) =
     let colArrays = m.ToColumnArrays()
     let mean = colArrays |> Array.map (fun x -> Statistics.Mean x)
     let std = colArrays |> Array.map (fun x -> Statistics.StandardDeviation x)
     m.MapIndexed (fun i j _ -> (m.At(i, j) - mean.[j]) / std.[j]), mean, std

let normalized, mean, std = featureNormalize X'

let X = week2.addIntercept normalized

let alpha = 0.01
let num_iters = 400

let initialTheta = vector [0.;0.;0.]    

let computeCostMulti (m : Matrix<double>) (v : Vector<double>) (t : Vector<double>) =
    let res = m * t - v
    let m = double res.Count
    1.0 / (2.0 * m)  * res.DotProduct(res)

let gradientDescentMulti (m: Matrix<double>) (v: Vector<double>) (t: Vector<double>) (a: float) iters =
    let costFct = Array.create iters 0.0
    let thetas = Array.create (iters + 1) (vector [0.; 0.])
    thetas.[0] <- t
    let updateTheta (theta': Vector<double>) = theta' - (m.Transpose() * (m * theta' - v) * (a / float v.Count))
    for i in 0 .. iters - 1 do
        let newTheta = updateTheta thetas.[i] 
        thetas.[i + 1] <- newTheta
        costFct.[i] <- computeCostMulti m v newTheta 
    thetas.[iters], costFct       


let optimalTheta, costs = gradientDescentMulti X y initialTheta alpha num_iters
optimalTheta

let _, costs2 = gradientDescentMulti X y initialTheta 0.1 50

let tmp = Array.zip [|0 .. 49|] costs2 |> Array.toList

Chart.Line tmp

    // //let gradientDescent (X: Matrix<double>) (y: Vector<double>) (alpha: float) (theta: Vector<double>) =
// //    theta - (X.Transpose() * (X * theta - y) * (alpha / float y.Count))

// //let numberIterations = 1500

// let optimalTheta_ (y: Vector<double>) (alpha: float) (initialTheta: Vector<double>) (X: Matrix<double>) =
//     [0..numberIterations] |> List.fold (fun parameters iter -> gradientDescent X y alpha parameters) initialTheta

// //let optimalTheta = m |> featureNormalize |> optimalTheta_ v 0.01 initialTheta

let normalEqn (m : Matrix<double>) (v : Vector<double>) =
    m.TransposeThisAndMultiply(m).Inverse() * m.TransposeThisAndMultiply(v)

let optimalTheta' = normalEqn X y 
optimalTheta'
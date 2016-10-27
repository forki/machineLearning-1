#load "../packages/MathNet.Numerics.FSharp.3.13.1/MathNet.Numerics.fsx"
#load "../packages/FSharp.Charting.Gtk.0.90.14/FSharp.Charting.Gtk.fsx"
#r "../packages/MathNet.Numerics.Data.Matlab.3.2.0/lib/net40/MathNet.Numerics.Data.Matlab.dll"

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Data.Matlab
open FSharp.Charting

let x = MatlabReader.Read<double>("/Users/carsten/Projects/courseramachinelearning/machine-learning-ex1/ex1/ex1data1.mat", "X")
let y' = MatlabReader.Read<double>("/Users/carsten/Projects/courseramachinelearning/machine-learning-ex1/ex1/ex1data1.mat", "y")

let y = y'.Column(0)

let initialTheta = vector [ 0.0 ; 0.0 ]
        
let cost (x : Matrix<double>) (y : Vector<double>) (theta : Vector<double>) = 
    let res = x * theta - y
    let m = double res.Count
    let norm = res.L2Norm()
    1.0 / (2.0 * m)  * norm * norm
                                               
cost x y initialTheta

//Chart.Point [ for i in 0 .. 96 -> x.At(i, 1), y.At(i) ]

let gradientDescent (X: Matrix<double>) (y: Vector<double>) (alpha: float) (theta: Vector<double>) =
    theta - (X.Transpose() * (X * theta - y) * (alpha / float y.Count))

let updateTheta theta = gradientDescent x y 0.01 theta

let numberIterations = 1500

let optimalTheta = [0..numberIterations] |> List.fold (fun parameters iter -> updateTheta parameters) initialTheta

cost x y optimalTheta

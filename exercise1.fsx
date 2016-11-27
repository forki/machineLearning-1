(**
Machine Learning Online Class - Exercise 1: Linear Regression
=============================================================

Initialization *)

module Exercise1

#load "packages/FsLab/FsLab.fsx"
#I "packages/MathNet.Numerics.Data.Text/lib/net40/"
#r "MathNet.Numerics.Data.Text.dll"

open System.IO
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Data.Text
open XPlot.GoogleCharts

(**
Part 1: Basic Function
----------------------
*)

let eye dim = DenseMatrix.identity<double> dim
let A = eye 5
(*** define-output:test ***)
A |> printfn "5x5 Identity Matrix: %A"
(*** include-output:test ***)

(**
Part 2: Plotting
----------------

Load data from text file using functionality from the MathNet.Numerics.Data.Text package.
*)

let fileName = Path.Combine(__SOURCE_DIRECTORY__, "ex1data1.txt")
let data = DelimitedReader.Read<double>(fileName, false, ",", false);
let X' = data.RemoveColumn(1)
let y = data.Column(1)

(** number of training examples *)
let m = y.Count

let scatterPoints = [ for i in 0 .. m - 1 -> X'.At(i, 0), y.At(i) ]
let options =
    Options(title = "Scatter plot of training data",
            hAxis = Axis(title = "Population of City in 10.000s", minValue = 4, maxValue = 24),
            vAxis = Axis(title = "Profit in $10.000s", minValue = -5, maxValue = 25),
            pointSize = 10,
            pointShape = "star")

(*** define-output:chart ***)
scatterPoints
     |> Chart.Scatter
     |> Chart.WithOptions options
(*** include-it:chart ***)

(**
Part 3: Gradient descent
------------------------

 Add a column of ones to X *)
let addIntercept (m : Matrix<double>) =
    let rows = m.RowCount
    let intercept = DenseVector.init rows (fun _ -> 1.0)
    m.InsertColumn(0, intercept)

let X = addIntercept X'

(** Initialize fitting parameters *)
let theta = vector [ 0.0 ; 0.0 ]

(** Some gradient descent settings *)
let iterations = 1500
let alpha = 0.01

(**
Cost function
*)
let computeCost (m : Matrix<double>) (v : Vector<double>) (t : Vector<double>) =
    let res = m * t - v
    let m = double res.Count
    let norm = res.L2Norm()
    1.0 / (2.0 * m)  * norm * norm


(**
Compute and display initial cost

Expected value of cost function for the initial fitting parameter is 32.07
*)
let cost = computeCost X y theta

(*** define-output:cost ***)
printfn "Cost: %f" cost
(*** include-output:cost ***)

let gradientDescent (m: Matrix<double>) (v: Vector<double>) (t: Vector<double>) (a: float) iters =
    let updateTheta (theta': Vector<double>) = theta' - (m.Transpose() * (m * theta' - v) * (a / float v.Count))
    [0..iters] |> List.fold (fun parameters iter -> updateTheta parameters) t


(** Run gradient descent *)
let optimalTheta = gradientDescent X y theta alpha iterations

(** Print theta to screen *)

(*** define-output:theta ***)
printfn "Theta found by gradient descent: %f, %f" (optimalTheta.At(0)) (optimalTheta.At(1))
(*** include-output:theta ***)

let regression x =
    let x' = vector [1.0; x]
    optimalTheta * x'

let regressionLine = [ for x in 0. .. 0.1 .. 25.0 -> x, regression x ]

let options2 =
    Options(title = "Training data with linear regression fit",
            hAxis = Axis(title = "Population of City in 10.000s", minValue = 4),
            vAxis = Axis(title = "Profit in $10.000s", minValue = -5),
            series = [|Series(``type`` = "scatter"); Series(``type`` = "line")|])

(*** define-output:chart2 ***)
[scatterPoints; regressionLine]
     |> Chart.Combo
     |> Chart.WithOptions options2
     |> Chart.WithLabels ["Training data"; "Regression Line"]
(*** include-it:chart2 ***)

(*** define-output:computeCost ***)
let optimalCost = computeCost X y optimalTheta
(*** include-output:computeCost ***)

(**
Predict values for population sizes of 35,000 and 70,000
*)
let predict1 = (vector [1.0; 3.5]) * optimalTheta

(*** define-output:theta1 ***)
printfn "For population = 35,000, we predict a profit of %f" (predict1 * 10000.0)
(*** include-output:theta1 ***)

let predict2 = (vector [1.0; 7.0]) * optimalTheta

(*** define-output:theta2 ***)
printfn "For population = 70,000, we predict a profit of %f" (predict2 * 10000.0)
(*** include-output:theta2 ***)


(**
Part 4: Visualizing J(theta_0, theta_1)
---------------------------------------
*)
let size = 100
let theta0_vals = Generate.LinearSpaced(size, -10.0, 10.0)
let theta1_vals = Generate.LinearSpaced(size, -1.0, 4.0)
let Jvals = [for i in theta0_vals -> [for j in theta1_vals -> computeCost X y (vector [float i; float j]) ]]

let foo = X

open XPlot.Plotly

let layout2 =
     Layout(
         title = "Surface",
         autosize = false,
         margin =
             Margin(
                 l = 65.,
                 r = 50.,
                 b = 65.,
                 t = 90.
             )
     )

Surface(z = Jvals)
 |> Chart.Plot
 |> Chart.WithLayout layout2
 |> Chart.WithWidth 700
 |> Chart.WithHeight 500

let z = Array2D.init size size (fun i j -> computeCost foo y (vector [ theta0_vals.[i]; theta1_vals.[j]]))


Contour(
    z = z,
    x = theta0_vals,
    y = theta1_vals
)
|> Chart.Plot
|> Chart.WithWidth 700
|> Chart.WithHeight 500



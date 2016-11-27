module Exercise2

#load "packages/FsLab/FsLab.fsx"

#I "packages/MathNet.Numerics.Data.Matlab/lib/net40/"
#I "packages/MathNet.Numerics.Data.Text/lib/net40/"
#r "MathNet.Numerics.Data.Matlab.dll"
#r "MathNet.Numerics.Data.Text.dll"

#r "packages/Accord/lib/net45/Accord.dll"
#r "packages/Accord.Math/lib/net45/Accord.Math.Core.dll"
#r "packages/Accord.Math/lib/net45/Accord.Math.dll"
#r "packages/Accord.Statistics/lib/net45/Accord.Statistics.dll"

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Data.Text
open MathNet.Numerics.Statistics;
open XPlot.GoogleCharts
open System.Linq
open System.IO
open System
open System.Globalization
open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting
open Accord.Math.Optimization

let fileName = Path.Combine(__SOURCE_DIRECTORY__, "ex2data1.txt")
let data = DelimitedReader.Read<double>(
            fileName, false, ",", true, new CultureInfo("en-US"))
let X = data.RemoveColumn(2)
let y = data.Column(2)

let pos = query { for res in data.EnumerateRows() do
                  where (res.At(2) = 1.0)
                  select (res.At(0), res.At(1)) } |> List.ofSeq

let neg = query { for res in data.EnumerateRows() do
                  where (res.At(2) = 0.0)
                  select (res.At(0), res.At(1)) } |> List.ofSeq

let options =
    Options(title = "Scatter plot of training data",
            hAxis = Axis(title = "Exam 1 score"), //, minValue = 4),
            vAxis = Axis(title = "Exam 2 score"), //, minValue = -5),
            series = [|Series(``type`` = "scatter"); Series(``type`` = "scatter")|])

[pos; neg]
     |> Chart.Combo
     |> Chart.WithOptions options
     |> Chart.WithLabels ["Admitted"; "Not admitted"]

type MathOps =
    static member Sigmoid(x:double) = 1.0 / (1.0 + Math.Exp(-x))
    static member Sigmoid(m:Matrix<double>) = m.Map (fun x -> MathOps.Sigmoid(x))
    static member Sigmoid(v:Vector<double>) = v.Map (fun x -> MathOps.Sigmoid(x))
    static member Log(v:Vector<double>) = v.Map (fun x -> Math.Log(x))

MathOps.Sigmoid 0.

let hypothesis (theta : Vector<double>) (x : Vector<double>) =
    MathOps.Sigmoid theta * x

let t = vector [3.0; 5.0]
let x = vector [2.0; 7.0]
hypothesis t x

let computeCost (m:Matrix<double>) (v:Vector<double>) (t:Vector<double>) =
    let m' = v.Count
    // (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta)))
    let tmp = -v * MathOps.Log (MathOps.Sigmoid (m * t)) + (v - 1.) * MathOps.Log (1.0 - MathOps.Sigmoid (m * t))
    tmp / (double m')

let gradient (m:Matrix<double>) (v:Vector<double>) (t:Vector<double>) =
    let m' = v.Count
    (1. / float m') * m.Transpose() * (MathOps.Sigmoid (m * t) - v)

let regression = new LogisticRegression()
regression.NumberOfInputs <- 2

let input = X.ToRowArrays()
let output = y |> Seq.toArray |> Array.map (fun g -> int g)


let learner = new IterativeReweightedLeastSquares(regression)
learner.Tolerance <- 1e-6  // Let's set some convergence parameters
learner.Iterations <- 400  // maximum number of iterations to perform
learner.Regularization <- 0.

let reg = learner.Learn(input, output)

let computeOutput = reg.Score ([| 45.; 85. |])

let addIntercept (m : Matrix<double>) =
    let rows = m.RowCount
    let intercept = DenseVector.init rows (fun _ -> 1.0)
    m.InsertColumn(0, intercept)

let X' = addIntercept X 

let compCost (t : double[]) =
    let theta = DenseVector.ofArray t
    computeCost X' y theta

let grad (t : double[]) =
    let theta = DenseVector.ofArray t
    let tmp = gradient X' y theta
    tmp.ToArray()

let f = new System.Func<double[], double>(compCost)

let g = new System.Func<double[], double[]>(grad)

let numberOfVariables = X'.ColumnCount
let lbfgs = new BroydenFletcherGoldfarbShanno(numberOfVariables, f, g)

let success = lbfgs.Minimize();
let minValue = lbfgs.Value;
let solution = lbfgs.Solution;

lbfgs.Status

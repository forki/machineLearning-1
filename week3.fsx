module Week3

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
open System.Linq
open System.IO
open System

let fileName = Path.Combine(__SOURCE_DIRECTORY__, "ex2data1.txt")

let data = DelimitedReader.Read<double>(
            fileName, false, ",", true)

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

#load "week2.fsx"
let XX = week2.addIntercept X

let computeCost (m:Matrix<double>) (v:Vector<double>) (t:Vector<double>) =
    let m' = v.Count
    // (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta)))
    let tmp = -v * MathOps.Log (MathOps.Sigmoid (m * t)) + (y - 1.) * MathOps.Log (1.0 - MathOps.Sigmoid (m * t))
    tmp / (float m')

let initialTheta = DenseVector.create XX.RowCount 0.0 

 

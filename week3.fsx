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



let data = DelimitedReader.Read<double>(
            "ex2data1.txt", false, ",", true)

let X = data.RemoveColumn(2)
let y = data.Column(2)

System.IO.Directory.GetCurrentDirectory()


let pos = data.EnumerateRows().ToList()


let bar = query { for res in pos do 
                  where (res.At(2) = 1.0)
                  select res }


let options =
    Options(title = "Scatter plot of training data",
            hAxis = Axis(title = "Exam 1 score"), //, minValue = 4),
            vAxis = Axis(title = "Exam 2 score"), //, minValue = -5),
            series = [|Series(``type`` = "scatter"); Series(``type`` = "scatter")|])

(*** define-output:chart2 ***)
[scatterPoints; regressionLine]
     |> Chart.Combo
     |> Chart.WithOptions options2
     |> Chart.WithLabels ["Training data"; "Regression Line"]
(*** include-it:chart2 ***)

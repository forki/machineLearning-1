module Exercise3

#load "exercise2.fsx"
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
open Exercise2

let computeRegularizedCost (X:Matrix<double>) (y:Vector<double>) (t:Vector<double>) l =
    let m = double y.Count
    let tmp = -y * MathOps.Log (MathOps.Sigmoid (X * t)) - (1. - y) * MathOps.Log (1. - MathOps.Sigmoid (X * t))
    let bar = (t.SubVector(1, t.Count)).L2Norm()
    (tmp + (l / 2.) * bar )  / m

let gradient (X:Matrix<double>) (y:Vector<double>) (t:Vector<double>) l =
    let m = double y.Count
    let t' = t.Clone()
    t'.At(0, 0.) 
    (1. / m) * X.Transpose() * (MathOps.Sigmoid (X * t) - y) + (l / m) * t'

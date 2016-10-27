#load "/Users/carsten/Tmp/machineLearning/packages/MathNet.Numerics.FSharp.3.13.1/MathNet.Numerics.fsx"
#r "/Users/carsten/Tmp/machineLearning/packages/MathNet.Numerics.Data.Matlab.3.2.0/lib/net40/MathNet.Numerics.Data.Matlab.dll"
#load "/Users/carsten/Tmp/machineLearning/packages/FSharp.Charting.Gtk.0.90.14/FSharp.Charting.Gtk.fsx"

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Data.Matlab
open FSharp.Charting

let x = MatlabReader.Read<double>("/Users/carsten/Projects/machinelearning/machine-learning-ex1/ex1/ex1data1.mat", "X")
let y' = MatlabReader.Read<double>("/Users/carsten/Projects/machinelearning/machine-learning-ex1/ex1/ex1data1.mat", "y")

let y = y'.Column(0)

let initialTheta = vector [ 0.0 ; 0.0 ]
        
let cost (theta : Vector<double>) (x : Matrix<double>) (y : Vector<double>) = 
    let res = x * theta - y
    let m = double res.Count
    let norm = res.L2Norm()
    1.0 / (2.0 * m)  * norm * norm
                                               
cost initialTheta x y

Chart.Point [ for i in 0 .. 96 -> x.At(i, 1), y.At(i) ]

let update (x : Matrix<double>) (y : Vector<double>) (theta : Vector<double>) alpha =
    let m = y.Count
    let mf = float m
    let n = x.ColumnCount
    let v = Vector<double>.Build.Dense(2)
    for j = 0 to n - 1 do
        let foo = [| for i in 0 .. m - 1 do yield (x.Row(i) * theta - y.[i]) * x.[i, j] |]
        let acc = foo |> Array.sum |> (fun x -> x * alpha / mf)
        do v.[j] <- theta.[j] - acc
    v

let gradientDescent (x : Matrix<double>) (y : Vector<double>) (theta : Vector<double>) alpha numiters =
    let mutable th = theta
    for i = 0 to numiters do
        let res = update x y th alpha
        th <- res
    th

let rec gradientDescent2 (x : Matrix<double>) (y : Vector<double>) (theta : Vector<double>) alpha numiters =
    let res = update x y theta alpha  
    let next = numiters - 1
    if numiters = -1 then theta 
    else gradientDescent2 x y (update x y theta alpha) alpha (numiters - 1)  


let newTheta = vector [0.0; 1.0]


update x y newTheta 0.01
let optTheta  = gradientDescent x y newTheta 0.01 1500
let optTheta2 = gradientDescent2 x y newTheta 0.01 1500


cost optTheta x y 
cost optTheta2 x y 


let gds_vec (X: Matrix<double>) (y: Vector<double>)  (α: float) (θ: Vector<double>) =
    let bar = X.Transpose()
    θ - (X.Transpose() * (X * θ - y) * (α / float y.Count))



let rec gradientDescent3 (x : Matrix<double>) (y : Vector<double>) (theta : Vector<double>) alpha numiters =
    let res = gds_vec x y  alpha theta  
    let next = numiters - 1
    if numiters = -1 then theta 
    else gradientDescent3 x y (update x y theta alpha) alpha (numiters - 1)  

let rec gradientDescent4 (x : Matrix<double>) (y : Vector<double>) alpha (theta : Vector<double>) = function 
    | 0 -> theta 
    | m -> gradientDescent4 x y alpha (gds_vec x y  alpha theta) (m - 1)


let rec gradientDescent5 (x : Matrix<double>) (y : Vector<double>) alpha numiters (theta : Vector<double>) =
    if numiters = -1 then theta 
    else gradientDescent5 x y alpha (numiters - 1) (update x y theta alpha)





let optTheta3 = gradientDescent3 x y newTheta 0.01 1500
cost optTheta3 x y 

let optTheta4 = gradientDescent4 x y 0.01 newTheta 1500
cost optTheta4 x y 

let optTheta5 = gradientDescent5 x y  0.01 1500 newTheta
cost optTheta5 x y 



let update2 t = gds_vec x y 0.01 t

update2 newTheta

[1..1500] |> List.fold (fun parameters iter -> update2 parameters) newTheta
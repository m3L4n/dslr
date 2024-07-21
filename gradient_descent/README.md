# Gradent descent
We will see the  gradent desent and two other implementaion of gradent descent
Gradent descent is a optimized algorithmn which calculate the derivate of the loss function  to get the minimun loss between prediction and the truth

* We need to choose the right learning rate 
* we need to choose the right n_umber of iterations

### How to know if my learning rate is good or if number of iterations is good 
I advice to print (plot) the loss function  at each iterations to see if our program learn right and  also to see if the number of iterations need to be adjust ( if the curves doesnt change at x iterations or if we doesnt iterate)


### How to choose the right gradent descent

- GD is more accurate but it provides a large complexity o(n_itr*  nbr_sample  * nbr_feature)
- SGD is less accurate but its complexity  is ok ( n_itr * n_features)
- Mini Batch i dont know at the moment how to calculate it ( TODO)

## Regular Gradent descent For Logistic regression
> its take the whole X at each iterations
 - Step 1 : choose Learning Rate and n_iterations
- Step 2 : Create the prediction of X with the weight so x = ( X * weight) for whole X 
 - Step 3 : create the sigmoid function  pred = ( 1 / 1 - exp^(-x)) that provides pred [0, 1] 
    - 0 -> not the class
    -   between 0.5 to 1 -> the class 
- Step 4 : Compute the gradient 1/ n_sample * ( Transpose de X * (pred  - the actual truth))
-  Step 5 : Update your weight: weight actuel -= lr * gradient 
-  reapet n_iteration 
-  step 6 ( optional) : calculate loss of the function and plot it to see if the lr and n_itr is good

## Stockastict Gradent descent
> its take one sample at each iterations
 - Step 1 : choose Learning Rate and n_iterations
 - Step 2 : Create the prediction of random x[i] from X  with the weight so x = ( X[i] * weight) 
 - Step 3: create the sigmoid function  pred = ( 1 / 1 - exp^(-x)) that provides pred [0, 1] 
    - 0 -> not the class
    -   between 0.5 to 1 -> the class 
- Step 4 : Compute SGD sgd =( x_i * (pred  - the actual truth))
- Step 5 : update your weight : weight -= lr * sgd
-  step 6 ( optional) : calculate loss of the function and plot it to see if the lr and n_itr is good
reapet n_iteration 

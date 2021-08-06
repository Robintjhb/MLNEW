import numpy as np

# y = wx + b
def compute_error_for_line_given_points(b, w, points): 
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2  #loss --单个loss
    return totalError / float(len(points))

#梯度信息
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))      #对b求导，loss
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))  #对w求导，loss
    new_b = b_current - (learningRate * b_gradient) #新的b=当前b-(学习率*b的梯度loss)
    new_w = w_current - (learningRate * w_gradient) #新的w
    return [new_b, new_w]

#执行循环
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]

def run():
    points = np.genfromtxt("data.csv", delimiter=",") #np读取csv文件
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess 初始化b
    initial_m = 0 # initial slope guess 初始化w
    num_iterations = 1000 #运行次数
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  compute_error_for_line_given_points(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations) #寻找最优点
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )

if __name__ == '__main__':
    run()
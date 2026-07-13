import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from scipy.integrate import solve_ivp

#Lorenz's parameters
sigma = 10
beta = 8/3
rho = 28

#########################################
# Functions
#########################################

#Function for the Lorenz Method 
def lorenz(t, y):

    dy = [sigma * (y[1] - y[0]), 
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2]]
    return np.array(dy)

#Function for the Euler step 
def euler_step(y0, T, dt):
    # Compute the total number of time steps
    steps = int(T / dt) #Time steps = Total time / time step
    X = np.zeros((3, steps + 1)) 
    X[:, 0] = y0 #initialize the first column X with the initial condition y0 with 0 steps + 1 columns for the time steps
    x = y0.copy() #separate block of memory for the current state x to avoid modifying the original initial condition y0

    # Forward Euler integration
    for k in range(1, steps + 1):
        x = x + dt * lorenz(0, x)  # Update the state using the Lorenz equations
        X[:, k] = x  # Store the updated state in the trajectory array

    t = np.arange(0, T + dt, dt)
    return X, t  # Return the trajectory and corresponding time points

#########################################
# Time Constraints and solving with Euler 
#########################################
# Initial conditions - change to visualize pertubations. 
y0 = np.array([1, 1, 1])      # Starting point in the Lorenz system

#Time Constraints 
dt = 0.001                      #time steps 
T = 100                      #total time 

num_time_pts = int(T / dt) + 1 # number of time points and shows hoa many time values there are from the start to the end of the simulation. 
t_eval = np.linspace(0, T, num_time_pts) # use linspace to create a time vector 

#Solve using Euler 
Y_euler, t_euler = euler_step(y0, T, dt) 
print("Euler first 10:" + str(Y_euler[:, :10])) 

#########################################
# Plotting 
#########################################

#Plot state evolution as a function of time 
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(t_euler, Y_euler[2, :], color='blue', label='Forward Euler')
ax.set_title('Lorenz-63 State Evolution - Forward Euler', fontsize=15)
ax.set_xlabel('Time')
ax.grid(True)

#Plot state evolution in interactive x, y, z, space. Figure 2
ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d')
ax.set_title('Euler Trajectory', fontsize=15)
ax.plot(Y_euler[0, :], Y_euler[1, :], Y_euler[2, :], color='blue', lw=0.5, label = 'Forward Euler')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')

#Create a clear background for the final result. Figure 3
ax.plot(
    Y_euler[0, :],
    Y_euler[1, :],
    Y_euler[2, :],
    color='blue',
    lw=0.5,
    label='Forward Euler'
)

# Remove grid
ax.grid(False)

# Remove the gray background panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Make pane edges invisible
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')

# Remove axis lines, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# Hide the axis frame
ax.set_axis_off()

fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=Y_euler[0, :],
        y=Y_euler[1, :],
        z=Y_euler[2, :],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Forward Euler'
    )
)

fig.update_layout(
    title='Euler Trajectory',
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        bgcolor='white'
    ),
    showlegend=False
)

fig.write_html("Euler_interactive.html") #Figure nunber 3

#########################################
# Animating the Plot
#########################################

# # Create figure and 3D axes
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# ax.set_title("Lorenz-63 Trajectory")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # Set axis limits 
# ax.set_xlim(np.min(Y_euler[0]), np.max(Y_euler[0]))
# ax.set_ylim(np.min(Y_euler[1]), np.max(Y_euler[1]))
# ax.set_zlim(np.min(Y_euler[2]), np.max(Y_euler[2]))

# # Create an empty line
# line, = ax.plot([], [], [], color='green', lw=1)

# # Animation function
# def animate(i):
#     line.set_data(Y_euler[0, :i], Y_euler[1, :i])
#     line.set_3d_properties(Y_euler[2, :i])
#     return line,

# # Create animation
# anim = animation.FuncAnimation(
#     fig,
#     animate,
#     frames=Y_euler.shape[1],
#     interval=10,
#     blit=False
# )

# plt.show()

# Plotting the Lorenz trajectory in an interactive 3D plot using Plotly. Figure 4
# Create figure
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Set title and labels
ax.set_title("Lorenz-63 Trajectory - Euler")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set fixed axis limits
ax.set_xlim(np.min(Y_euler[0]), np.max(Y_euler[0]))
ax.set_ylim(np.min(Y_euler[1]), np.max(Y_euler[1]))
ax.set_zlim(np.min(Y_euler[2]), np.max(Y_euler[2]))

# Create empty trajectory and point
trajectory, = ax.plot([], [], [], color='blue', lw=1, label='Trajectory')
point, = ax.plot([], [], [], 'ro', markersize=4, color = 'blue', label='Current State')

ax.grid(False)

# Animation function
def animate(i):

    # Draw trajectory up to current point
    trajectory.set_data(
        Y_euler[0, :i],
        Y_euler[1, :i]
    )
    trajectory.set_3d_properties(
        Y_euler[2, :i]
    )

    # Draw moving point
    point.set_data(
        [Y_euler[0, i]],
        [Y_euler[1, i]]
    )
    point.set_3d_properties(
        [Y_euler[2, i]]
    )

    return trajectory, point

step = 15

#Create animation
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=range(0, Y_euler.shape[1], step), #shape -> # of colums in Euler 
    interval=20,                             # playback speed 
)

# anim.save(
#     "Euler.mp4",
#     writer="ffmpeg",
#     fps=50
# )

plt.show(block = True)


#########################################
# Solving with RK4
#########################################

# Runge-Kutta 4th order method for solving ODEs
def rk4_step(fun, dt, t0, y0):
    f1 = fun(t0, y0)    #t0 equivalent to Xk
    f2 = fun(t0 + dt/2, y0 + dt/2 * f1)
    f3 = fun(t0 + dt/2, y0 + dt/2 * f2)
    f4 = fun(t0 + dt, y0 + dt * f3)
    yout = y0 + (dt/6) * (f1 + 2*f2 + 2*f3 + f4)
    return yout


#Solve using rk4
Y = np.zeros((3, num_time_pts)) # Y array holds the state variables of Xk through each step of the RK4 integration. 
Y[:, 0] = y0                    # Starting with y0 as the initial condition for the first column of Y. Continues to fill the Y array with the state variables at each time step.
yin = y0
for i in range (num_time_pts - 1):
    yout = rk4_step(lorenz, dt, t_eval[i], yin)
    Y[:, i + 1] = yout
    yin = yout
print("R4K first 10:" + str(Y[:, :10]))

#########################################
# Plotting 
#########################################

#Plot state evolution as a function of time 
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(t_eval, Y[2, :], color='blue', label='RK4')
ax.set_title('Lorenz-63 State Evolution - RK4', fontsize=15)
ax.set_xlabel('Time')
ax.grid(True)

#Plot state evolution in interactive x, y, z, space. 
ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d')
ax.set_title('RK4 Trajectory', fontsize=15)
ax.plot(Y[0, :], Y[1, :], Y[2, :], color='blue', lw=0.5, label = 'RK4')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')

#Create a clear background for the final result. 
ax.plot(
    Y_euler[0, :],
    Y_euler[1, :],
    Y_euler[2, :],
    color='blue',
    lw=0.5,
    label='Forward Euler'
)

# Remove grid
ax.grid(False)

# Remove the gray background panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Make pane edges invisible
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')

# Remove axis lines, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# Hide the axis frame
ax.set_axis_off()


fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=Y[0, :],
        y=Y[1, :],
        z=Y[2, :],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Forward Euler'
    )
)

fig.update_layout(
    title='RK4 Trajectory',
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        bgcolor='white'
    ),
    showlegend=False
)

fig.write_html("RK4_interactive.html")

# Create figure
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Set title and labels
ax.set_title("Lorenz-63 Trajectory - RK4")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set fixed axis limits
ax.set_xlim(np.min(Y_euler[0]), np.max(Y_euler[0]))
ax.set_ylim(np.min(Y_euler[1]), np.max(Y_euler[1]))
ax.set_zlim(np.min(Y_euler[2]), np.max(Y_euler[2]))

# Create empty trajectory and point
trajectory, = ax.plot([], [], [], color='blue', lw=1, label='Trajectory')
point, = ax.plot([], [], [], 'ro', markersize=4, color = 'blue', label='Current State')

ax.grid(False)


#########################################
# Animating the Plot
#########################################

# Animation function
def animate(i):

    # Draw trajectory up to current point
    trajectory.set_data(
        Y_euler[0, :i],
        Y_euler[1, :i]
    )
    trajectory.set_3d_properties(
        Y_euler[2, :i]
    )

    # Draw moving point
    point.set_data(
        [Y_euler[0, i]],
        [Y_euler[1, i]]
    )
    point.set_3d_properties(
        [Y_euler[2, i]]
    )

    return trajectory, point

step = 15

#Create animation
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=range(0, Y_euler.shape[1], step), #shape -> # of colums in Euler 
    interval=20,                             # playback speed 
)

# anim.save(
#     "RK4.mp4",
#     writer="ffmpeg",
#     fps=50
# )

plt.show(block = True)

#########################################
# Solving with IVP
#########################################

# Solve using solve_ivp
lorenz_solution = solve_ivp(
    lorenz,
    [0, T],
    y0,
    t_eval=t_eval,
    method='RK45'
)
print("solve_ivp first 10:" + str(lorenz_solution.y[:, :10]))

t_ivp = lorenz_solution.t
y_ivp = lorenz_solution.y

#########################################
# Plotting 
#########################################

#Plot state evolution as a function of time 
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(t_ivp, y_ivp[2, :], color='blue', label='solve_ivp')
ax.set_title('Lorenz-63 State Evolution - solve_ivp', fontsize=15)
ax.set_xlabel('Time')
ax.grid(True)

#Plot state evolution in interactive x, y, z, space. 
ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d')
ax.set_title('solve_ivp Trajectory', fontsize=15)
ax.plot(y_ivp[0, :], y_ivp[1, :], y_ivp[2, :], color='blue', lw=0.5, label = 'solve_ivp')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')

#Create a clear background for the final result. 
ax.plot(
    y_ivp[0, :],
    y_ivp[1, :],
    y_ivp[2, :],
    color='blue',
    lw=0.5,
    label='solve_ivp'
)

# Remove grid
ax.grid(False)

# Remove the gray background panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Make pane edges invisible
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')

# Remove axis lines, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# Hide the axis frame
ax.set_axis_off()


fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=y_ivp[0, :],
        y=y_ivp[1, :],
        z=y_ivp[2, :],
        mode='lines',
        line=dict(color='blue', width=2),
        name='solve_ivp'
    )
)

fig.update_layout(
    title='IVP Trajectory',
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        bgcolor='white'
    ),
    showlegend=False
)

fig.write_html("IVP_interactive.html")

# Create figure
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Set title and labels
ax.set_title("Lorenz-63 Trajectory - solve_ivp")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set fixed axis limits
ax.set_xlim(np.min(y_ivp[0]), np.max(y_ivp[0]))
ax.set_ylim(np.min(y_ivp[1]), np.max(y_ivp[1]))
ax.set_zlim(np.min(y_ivp[2]), np.max(y_ivp[2]))

# Create empty trajectory and point
trajectory, = ax.plot([], [], [], color='blue', lw=1, label='Trajectory')
point, = ax.plot([], [], [], 'ro', markersize=4, color = 'blue', label='Current State')

ax.grid(False)


#########################################
# Animating the Plot
#########################################

# Animation function
def animate(i):

    # Draw trajectory up to current point
    trajectory.set_data(
        y_ivp[0, :i],
        y_ivp[1, :i]
    )
    trajectory.set_3d_properties(
        y_ivp[2, :i]
    )

    # Draw moving point
    point.set_data(
        [y_ivp[0, i]],
        [y_ivp[1, i]]
    )
    point.set_3d_properties(
        [y_ivp[2, i]]
    )

    return trajectory, point

step = 15

#Create animation
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=range(0, y_ivp.shape[1], step), #shape -> # of colums in Euler 
    interval=20,                             # playback speed 
)

# anim.save(
#     "IVP.mp4",
#     writer="ffmpeg",
#     fps=50
# )

plt.show(block = True)



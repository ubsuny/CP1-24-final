import matplotlib.pyplot as plt
import distance_calc as dc
import direction_of_motion as md
import final as fin
import numpy as np

def this():
    names=fin.md_list("/workspaces/CP1-24-final/abruns123/data/sin_data", ".csv")
    for n in names:
        x_pos,y_pos=fin.get_sin_data("/workspaces/CP1-24-final/abruns123/data/sin_data/"+n)
        y_pos=fin.subtract_ave(y_pos)
        par, func,x =fin.gauss_newton(np.array(x_pos),y_pos, [15, 1.2,0],10)
        plt.figure()
        plt.plot(x_pos, y_pos)
        plt.plot(x, func)
        plt.grid()
        plt.show()
        new_name=n.rstrip(".csv")
        plt.savefig("/workspaces/CP1-24-final/abruns123/data/sin_data/"+new_name+".png", format="png")
        plt.close()

def individual_fitter(path, name, p1, p2, p3):
    x_pos,y_pos=fin.get_sin_data(path+"/"+name)
    y_pos=fin.subtract_ave(y_pos)
    par, func,x =fin.gauss_newton(np.array(x_pos),y_pos, [p1, p2,p3],10)
    plt.figure()
    plt.plot(x_pos, y_pos)
    plt.plot(x, func)
    plt.grid()
    plt.show()
    new_name=name.rstrip(".csv")
    plt.savefig(path+"/"+new_name+".png", format="png")
    plt.close()

individual_fitter("/workspaces/CP1-24-final/abruns123/data/sin_data", "experiment_20.csv", 15, .62, eval("4*np.pi/3"))
this1="5*np.pi/6"
this2=eval(this1)


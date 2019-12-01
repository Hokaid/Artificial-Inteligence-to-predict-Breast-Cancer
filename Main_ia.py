#Importaciones:
import pandas as pd
import numpy as np
import scipy as sc
import sklearn
import mglearn
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import os.path

#Declaración de la ventana tkinter, tamaño y nombre
root = Tk()
root.geometry("600x410")
root.title("Red Neuronal con PCA para predecir Cáncer de seno")
root.config(bg="yellow")
#Creamos la imagen
fondo = PhotoImage(file="fondo.gif")
lblFondo = Label(root, image=fondo).place(x=0,y=0)

#Variables de tkinter:
filen = StringVar(value="") #Para guardar la ruta del archivo
trainp = IntVar() #Para guardar el porcentaje de datos de entrenamiento
nnodos = StringVar() #Cantidad de nodos por capa
maxit = IntVar() #Numero maximo de iteraciones
activ = StringVar(value="relu") #Para especificar función de activación
solv = StringVar(value="adam") #Para especificar el metodo para la optimización de pesos.
pcaok = StringVar(value="No") #Para especificar si se ejecutará PCA o no
preciB = IntVar() #precisión para Benignas
preciM = IntVar() #precisión para Malignas

#Funciones:

#Función para saber si un numero es un entero:
def es_entero(variable):
   try:
      int(variable)
      return True
   except:
      return False

#Función para obtener la ruta de un archivo
def abrir():
   ruta=askopenfilename()
   filen.set(ruta)

#Función para graficar en 3D
def hacer_grafico(data_pca, data_salida):
   arr_salida = np.zeros(569).reshape(569, 1)
   for k in range(569):
      arr_salida[k] = data_salida['diagnosis'][k]
   X = data_pca
   Y = arr_salida
   # Creamos la figura
   fig = plt.figure()
   #Creamos el plano 3D
   ax1 = fig.add_subplot(111, projection = '3d')
   #Graficar data resultante
   ax1.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], X[Y[:,0] == 0, 1], c="skyblue")
   ax1.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], X[Y[:,0] == 1, 1], c="salmon")
   ax1.legend(['malignat', 'benign'])
   plt.show()

def slice_data():
   data_pca = []; data_salida = []
   extension = os.path.splitext(filen.get())[1] #Se obtiene la extensión del archivo
   if (extension == ".csv"): #Se comprueba si la extensión es csv
      mamd = pd.read_csv(filen.get())
   else: #Se lanza mensaje de error sino es csv
      messagebox.showwarning(message="La extensión del archivo seleccionado debe ser igual a '.csv'.", title="Extensión incorrecta")
      return
   if (trainp.get() < 100 and trainp.get() > 2): #Se comprueba si el porcentaje de entrenamiento es adecuado
      tsize = 1 - (trainp.get()/100)
   else: #Se lanza mensaje de error sino es así
      messagebox.showwarning(message="El porcentaje de datos de entrenamiento debe ser menor a 100% y mayor a 2%.", title="Porcentaje incorrecto")
      return
   #Convertir dataset (variable diagnostico) de texto a numeros:
   mamd['diagnosis'] = mamd['diagnosis'].replace({"M": 0, "B": 1})
   pd.to_numeric(mamd['diagnosis'])
   if (pcaok.get() == "Sí"):
      data_entrada = mamd.filter(['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                                  'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean' ]) #Se filtran datos
      pca = PCA(n_components = 0.9999, svd_solver = 'full') #Declaración de PCA
      pca.fit(data_entrada) #Entrenar al algoritmo PCA con nuestra data
      data_pca = pca.transform(data_entrada) #Transformar la data
      data_salida = mamd.filter(['diagnosis']) #designacion de data diagnostico
      data = pd.DataFrame(data_pca) #Se convierte a Pandas por conveniencia
      data['diagnosis'] = data_salida #Se agrega la columna Clase
      #Se divide la data de entrenamieno y la de prueba:
      training_set, validation_set = train_test_split(data, test_size = tsize, random_state = 21)
      X_train = training_set.iloc[:,0:-1].values
      Y_train = training_set.iloc[:,-1].values
      X_val = validation_set.iloc[:,0:-1].values
      y_val = validation_set.iloc[:,-1].values
   elif (pcaok.get() == "No"):
      #Se divide la data de entrenamieno y la de prueba:
      training_set, validation_set = train_test_split(mamd, test_size = tsize, random_state = 21)
      X_train = training_set.iloc[:,2:12].values
      Y_train = training_set.iloc[:,1].values
      X_val = validation_set.iloc[:,2:12].values
      y_val = validation_set.iloc[:,1].values
   return X_train, Y_train, X_val, y_val, data_pca, data_salida
   
#Funcion que se llama cuando se apreta el boton para entrenar la red neuronal y probarla
def entrenar_probar():
   X_train, Y_train, X_val, y_val, data_pca, data_salida = slice_data()
   stnods = nnodos.get()
   hlsnodos = []
   for x in stnods.split(','): #Se comprueba que las cantidades de nodos por capa sean enteros
      if es_entero(x):
         hlsnodos.append(int(x))
      else: #Se lanza mensaje de error sino es así
         messagebox.showwarning(message="La cantidad de nodos por cada capa debe ser un numero entero.", title="Información incorrecta")
         return
   if (maxit.get() < 1000000 and maxit.get() > 10): #Se comprueba si el numero maximo de iteraciones es adecuado
      #Se crea y establece la red neuronal (MLP):
      classifier = MLPClassifier(hidden_layer_sizes=hlsnodos, max_iter=maxit.get(),activation = activ.get(),solver=solv.get(),random_state=1)
   else: #Se lanza mensaje de error sino es así
      messagebox.showwarning(message="El numero maximo de iteraciones debe ser menor a 1000000 y mayor a 10", title="Numero incorrecto")
      return
   classifier.fit(X_train, Y_train) #Se entrena el MLP
   y_pred = classifier.predict(X_val) #Se prueba el MLP
   cm = confusion_matrix(y_pred, y_val) #Se crea la matriz de confusión
   preciB.set((cm[0][0]/(cm[0][0]+cm[1][0]))*100) #Se actualizan datos para ser mostrados en pantalla
   preciM.set((cm[1][1]/(cm[0][1]+cm[1][1]))*100)
   if (pcaok.get() == "Sí"): hacer_grafico(data_pca, data_salida)
   

#Mensaje en Tkinter de selección de archivos, boton y caja de texto respectiva
mensArchivo = Label(root, text="Selecciona el archivo con los datos", background="coral")
mensArchivo.place(x=30, y=20)
entryArchivo = Entry(root, textvariable=filen, width=70)
entryArchivo.place(x=30, y=50)
botonAbrirArchivo =Button(root,text="Seleccionar archivo", command=abrir)
botonAbrirArchivo.place(x=470, y=48)

#Mensajes en Tkinter para ingreso de datos afines al entrenamiento y prueba de la red neuronal, y cajas de texto respectivas
mensTrain = Label(root, text="Ingrese el porcentaje de datos de entrenamiento (%): ", background="pink")
mensTrain.place(x=30, y=100)
entryTrain = Entry(root, textvariable=trainp, width=10)
entryTrain.place(x=500, y=100)
mensNnodos = Label(root, text="Ingrese el numero de nodos por capa ordenadamente separados por comas: ", background="coral")
mensNnodos.place(x=30, y=130)
entryNnodos = Entry(root, textvariable=nnodos, width=60)
entryNnodos.place(x=33, y=155)
mensMaxit = Label(root, text="Ingrese el numero maximo de iteraciones: ", background="pink")
mensMaxit.place(x=30, y=180)
entryMaxit = Entry(root, textvariable=maxit, width=10)
entryMaxit.place(x=500, y=180)

#Cajas de selección y Labels, para seleccionar función de activación, metodo de optimización y aplicación de PCA
mensActive = Label(root, text="Función de activación: ", background="coral")
mensActive.place(x=30, y=220)
combActive = ttk.Combobox(root, values=["relu", "logistic", "tanh", "identity"], state='readonly', textvariable=activ)
combActive.place(x=30, y=250)
mensSolver = Label(root, text="Optimización de pesos: ", background="pink")
mensSolver.place(x=200, y=220)
combSolver = ttk.Combobox(root, values=["adam", "lbfgs", "sgd"], state='readonly', textvariable=solv)
combSolver.place(x=200, y=250)
menPCA = Label(root, text="Ejecución de PCA: ", background="coral")
menPCA.place(x=375, y=220)
entryPCA = ttk.Combobox(root, values=["Sí", "No"], state='readonly', textvariable=pcaok)
entryPCA.place(x=375, y=250)

#Botones que llaman a la función entrenar_probar
botonEntrenar =Button(root,text="Entrenar y Probar", command=entrenar_probar)
botonEntrenar.place(x=30, y=300)

#Labels y cajas de texto que dan información sobre la precisión de diagnostico de la red neuronal
mporceBeni = Label(root, text="Porcentaje de precisión para diagnosticar un tumor benigno (%): ", background="coral")
mporceBeni.place(x=30, y=340)
entrympBeni = Entry(root, textvariable=preciB, width=10)
entrympBeni.place(x=400, y=340)
mporceMali = Label(root, text="Porcentaje de precisión para diagnosticar un tumor maligno (%): ", background="pink")
mporceMali.place(x=30, y=370)
entrympMali = Entry(root, textvariable=preciM, width=10)
entrympMali.place(x=400, y=370)


#Loop de Tkinter
root.mainloop()

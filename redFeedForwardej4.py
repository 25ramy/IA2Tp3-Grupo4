import numpy as np
import matplotlib.pyplot as plt

# Generador basado en ejemplo del curso CS231 de Stanford: 
# CS231n Convolutional Neural Networks for Visual Recognition
# (https://cs231n.github.io/neural-networks-case-study/)
def generar_datos_clasificacion(cantidad_ejemplos, cantidad_clases):
    FACTOR_ANGULO = 0.40
    AMPLITUD_ALEATORIEDAD = 0.3

    n = int(cantidad_ejemplos / cantidad_clases)
    x = np.zeros((cantidad_ejemplos, 2))
    t = np.zeros(cantidad_ejemplos, dtype="uint8")  # 1 columna: la clase correspondiente (t -> "target")

    randomgen = np.random.default_rng()
    for clase in range(cantidad_clases):
        radios = np.linspace(0, 1, n) + AMPLITUD_ALEATORIEDAD * randomgen.standard_normal(size=n)
        angulos = np.linspace(clase * np.pi * FACTOR_ANGULO, (clase + 1) * np.pi * FACTOR_ANGULO, n)

        indices = range(clase * n, (clase + 1) * n)
        x1 = radios * np.tanh(angulos)
        x2 = radios * np.cos(angulos)

        x[indices] = np.c_[x1, x2]

        t[indices] = clase

    return x, t


def inicializar_pesos(n_entrada, n_capa_2, n_capa_3):
    randomgen = np.random.default_rng()

    w1 = 0.1 * randomgen.standard_normal((n_entrada, n_capa_2))
    b1 = 0.1 * randomgen.standard_normal((1, n_capa_2))

    w2 = 0.1 * randomgen.standard_normal((n_capa_2, n_capa_3))
    b2 = 0.1 * randomgen.standard_normal((1,n_capa_3))

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def ejecutar_adelante(x, pesos):
    # Funcion de entrada (a.k.a. "regla de propagacion") para la primera capa oculta
    z = x.dot(pesos["w1"]) + pesos["b1"]

    # Funcion de activacion ReLU para la capa oculta (h -> "hidden")
    h = np.maximum(0, z)

    # Salida de la red (funcion de activacion lineal). Esto incluye la salida de todas
    # las neuronas y para todos los ejemplos proporcionados
    y = h.dot(pesos["w2"]) + pesos["b2"]

    return {"z": z, "h": h, "y": y}


def clasificar(x, pesos):
    # Corremos la red "hacia adelante"
    resultados_feed_forward = ejecutar_adelante(x, pesos)
    
    # Buscamos la(s) clase(s) con scores mas altos (en caso de que haya mas de una con 
    # el mismo score estas podrian ser varias). Dado que se puede ejecutar en batch (x 
    # podria contener varios ejemplos), buscamos los maximos a lo largo del axis=1 
    # (es decir, por filas)
    max_scores = np.argmax(resultados_feed_forward["y"], axis=1)

    # Tomamos el primero de los maximos (podria usarse otro criterio, como ser eleccion aleatoria)
    # Nuevamente, dado que max_scores puede contener varios renglones (uno por cada ejemplo),
    # retornamos la primera columna
    return max_scores

# x: n entradas para cada uno de los m ejemplos(nxm)
# t: salida correcta (target) para cada uno de los m ejemplos (m x 1)
# pesos: pesos (W y b)
def train(x, t, pesos, learning_rate, epochs,validacionx, validaciont,lossValAnt,condValidacion):
    if (condValidacion==True):
        #x ejemplos t sal deseada, pesos sinapticos (ini peso) learning rate ritmo aprendizaje, epoch 
        # Cantidad de filas (i.e. cantidad de ejemplos)
        m = np.size(x, 0) 
        lossValGraf=[]
        epochGraf=[]
        for i in range(epochs):
            # Ejecucion de la red hacia adelante
            resultados_feed_forward = ejecutar_adelante(x, pesos)
            y = resultados_feed_forward["y"] #val salida
            h = resultados_feed_forward["h"] #val capa oculta
            z = resultados_feed_forward["z"] #val entrada capa oculta

            # LOSS
            # a. Exponencial de todos los scores
            exp_scores = np.exp(y)

            # b. Suma de todos los exponenciales de los scores, fila por fila (ejemplo por ejemplo).
            #    Mantenemos las dimensiones (indicamos a NumPy que mantenga la segunda dimension del
            #    arreglo, aunque sea una sola columna, para permitir el broadcast correcto en operaciones
            #    subsiguientes)
            sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)

            # c. "Probabilidades": normalizacion de las exponenciales del score de cada clase (dividiendo por 
            #    la suma de exponenciales de todos los scores), fila por fila
            p = exp_scores / sum_exp_scores

            # d. Calculo de la funcion de perdida global. Solo se usa la probabilidad de la clase correcta, 
            #    que tomamos del array t ("target")
            loss = (1 / m) * np.sum( -np.log( p[range(m), t] ))
            # Mostramos solo cada 1000 epochs
            if i %500 == 0:
                print("Training Loss epoch", i, ":", loss)
                lossVal=validacion(validacionx, validaciont,pesos)
                epochGraf.append(i)
                lossValGraf.append(lossVal)
                if (lossVal>lossValAnt):
                    condValidacion=False
                    fig, ax = plt.subplots()
                    ax.plot(epochGraf, lossValGraf)

                    ax.set(xlabel='epoch', ylabel='Loss Val', title='Epoch//Loss Validation')
                    ax.grid()
                    plt.show()
                    break
                else:
                    lossValAnt=lossVal

            #medir accuracy,  aciertos/cant eje; loss medir con ejemplos test. no estan ni en training ni barrido de par
            # Extraemos los pesos a variables locales
            w1 = pesos["w1"]
            b1 = pesos["b1"]
            w2 = pesos["w2"]
            b2 = pesos["b2"]

            # Ajustamos los pesos: Backpropagation
            dL_dy = p                # Para todas las salidas, L' = p (la probabilidad)...
            dL_dy[range(m), t] -= 1  # ... excepto para la clase correcta
            dL_dy /= m

            dL_dw2 = h.T.dot(dL_dy)                         # Ajuste para w2
            dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)   # Ajuste para b2

            dL_dh = dL_dy.dot(w2.T)
            
            dL_dz = dL_dh       # El calculo dL/dz = dL/dh * dh/dz. La funcion "h" es la funcion de activacion de la capa oculta,
            dL_dz[z <= 0] = 0   # para la que usamos ReLU. La derivada de la funcion ReLU: 1(z > 0) (0 en otro caso)

            dL_dw1 = x.T.dot(dL_dz)                         # Ajuste para w1
            dL_db1 = np.sum(dL_dz, axis=0, keepdims=True)   # Ajuste para b1

            # Aplicamos el ajuste a los pesos
            w1 += -learning_rate * dL_dw1
            b1 += -learning_rate * dL_db1
            w2 += -learning_rate * dL_dw2
            b2 += -learning_rate * dL_db2

            # Actualizamos la estructura de pesos
            # Extraemos los pesos a variables locales
            pesos["w1"] = w1
            pesos["b1"] = b1
            pesos["w2"] = w2
            pesos["b2"] = b2

    return (pesos,lossValAnt,condValidacion)


def iniciar(numero_clases, numero_ejemplos, graficar_datos,validacionx, validaciont):
    # Generamos datos
    x, t = generar_datos_clasificacion(numero_ejemplos, numero_clases)

    
    # Graficamos los datos si es necesario
    if graficar_datos:
        # Parametro: "c": color (un color distinto para cada clase en t)
        plt.scatter(x[:, 0], x[:, 1], c=t)
        plt.show()

    # Inicializa pesos de la red
    NEURONAS_CAPA_OCULTA = 100
    NEURONAS_ENTRADA = 2
    pesos = inicializar_pesos(n_entrada=NEURONAS_ENTRADA, n_capa_2=NEURONAS_CAPA_OCULTA, n_capa_3=numero_clases)

    

    # Entrena
    LEARNING_RATE=1
    EPOCHS=10000 #2001
    lossValAnt=1000
    condValidacion=True
    (pesos_train,lossValAnt,condValidacion)=train(x, t, pesos, LEARNING_RATE, EPOCHS,validacionx, validaciont,lossValAnt,condValidacion)
    return pesos_train
#TEST
def test():
    x2, t2 = generar_datos_clasificacion(num_test, num_class)
    scoreTest= clasificar(x2,pesos)
    totalt= np.size(t2) #cantidad total de ejemplos clasificados como c
    correctos=0.00
    for i in range(0,totalt):
        if scoreTest[i]==t2[i]:
            correctos+=1
    precision=correctos/totalt
    print("Precision: ")
    print(precision)
    m = np.size(x2, 0) 

    resultados_feed_forward = ejecutar_adelante(x2, pesos)
    y = resultados_feed_forward["y"] #val salida
    #h = resultados_feed_forward["h"] #val capa oculta
    #z = resultados_feed_forward["z"] #val entrada capa oculta
    exp_scores = np.exp(y)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    p = exp_scores / sum_exp_scores

    loss = (1 / m) * np.sum( -np.log( p[range(m), t2] ))
    print("Test Loss epoch", ":", loss)

def validacion(x2,t2,pesos):
    m = np.size(x2, 0) 

    resultados_feed_forward = ejecutar_adelante(x2, pesos)
    y = resultados_feed_forward["y"] #val salida
    exp_scores = np.exp(y)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    p = exp_scores / sum_exp_scores

    loss = (1 / m) * np.sum( -np.log( p[range(m), t2] ))
    print("validacion Loss epoch", ":", loss)
    return loss


num_class=5
num_ej=300
num_test=30
num_val=30
validacionx, validaciont = generar_datos_clasificacion(num_val, num_class)
pesos=iniciar(num_class, num_ej, True, validacionx, validaciont) 
test()


# **Implementación de Redes Neuronales Convolucionales (CNNs) con PyTorch**

Este repositorio contiene las instrucciones para crear un cuaderno interactivo en Python que implemente una Red Neuronal Convolucional (CNN) utilizando PyTorch. Este ejercicio permite a los estudiantes familiarizarse con conceptos fundamentales de las CNNs, como la preparación de datos, el diseño de arquitectura, el entrenamiento y la evaluación.

## **Requisitos**

Antes de comenzar, asegúrate de tener instalados los siguientes requisitos en tu entorno:

- Python 3.8 o superior
- PyTorch
- Matplotlib
- Torchvision
- Scikit-learn (para métricas adicionales)

## **Pasos a seguir**

### **1. Preparación de datos**
1. Descarga un conjunto de datos de imágenes (por ejemplo, CIFAR-10 o MNIST).
2. Usa PyTorch para cargar y preprocesar los datos utilizando `torchvision.datasets` y `torch.utils.data.DataLoader`.
3. Divide el conjunto de datos en entrenamiento, validación y prueba.

### **2. Diseño de la arquitectura CNN**
1. Crea una clase que herede de `torch.nn.Module`.
2. Define una arquitectura CNN que incluya:
   - Capas convolucionales (`torch.nn.Conv2d`).
   - Capas de pooling (e.g., `torch.nn.MaxPool2d`).
   - Capas completamente conectadas (`torch.nn.Linear`).
   - Funciones de activación (e.g., `torch.nn.ReLU`).
3. Implementa un método `forward` para definir el flujo de datos a través de la red.

### **3. Entrenamiento del modelo**
1. Configura la función de pérdida (e.g., `torch.nn.CrossEntropyLoss`) y un optimizador (e.g., `torch.optim.SGD`).
2. Implementa un ciclo de entrenamiento que:
   - Realice el forward pass.
   - Calcule la pérdida.
   - Actualice los parámetros del modelo mediante backpropagation.
3. Monitorea las métricas de entrenamiento, como la pérdida y la precisión.

### **4. Evaluación del modelo**
1. Evalúa el rendimiento del modelo en el conjunto de validación.
2. Calcula métricas como:
   - Precisión
   - Recall
   - F1-score
3. Visualiza los resultados del entrenamiento y evaluación (e.g., gráficas de pérdida y precisión).

### **5. Pruebas finales**
1. Usa el conjunto de prueba para evaluar la capacidad de generalización del modelo.
2. Reporta las métricas obtenidas en el test.

## **Estructura sugerida del cuaderno interactivo**
El cuaderno debe incluir:
1. **Introducción:** Explicación de las CNNs y el objetivo del ejercicio.
2. **Preparación de datos:** Código para cargar y preprocesar los datos.
3. **Diseño de la CNN:** Definición de la arquitectura y su visualización.
4. **Entrenamiento:** Código para entrenar el modelo con visualización de métricas.
5. **Evaluación y pruebas:** Análisis del rendimiento y predicciones del modelo.
6. **Conclusión:** Reflexión sobre los resultados y posibles mejoras.

## **Entrega**
- **Formato:** Jupyter Notebook (.ipynb) o script Python (.py).
- Incluye comentarios y explicaciones claras en el código.
- Agrega visualizaciones de resultados (e.g., gráficas de pérdida, precisión y predicciones).
- Indica el dataset utilizado y cualquier recurso externo.

## **Criterios de evaluación**
Tu trabajo será evaluado en función de:
- **Carga y preprocesamiento de datos.**
- **Diseño e implementación de la arquitectura CNN.**
- **Entrenamiento y evaluación del modelo.**
- **Claridad y calidad del código y comentarios.**

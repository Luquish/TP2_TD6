# tools.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libc.time cimport time, strftime

def agrupar_categorias_cython(list categorical_features, list columns_to_exclude, list data, int umbral=100):
    """
    Agrupa categorías raras en 'Otro' y reemplaza NaN por 'Desconocido' usando Cython para mejorar el rendimiento.

    Parámetros:
    - categorical_features (list): Lista de nombres de columnas categóricas.
    - columns_to_exclude (list): Lista de nombres de columnas a excluir.
    - data (list of lists): Matriz de datos del DataFrame.
    - umbral (int): Umbral para considerar una categoría como rara.

    Retorna:
    - data (list of lists): Matriz de datos modificada.
    """
    cdef int col, row
    cdef int n_cols = len(categorical_features)
    cdef int n_rows = len(data)
    cdef object category
    cdef dict frecuencia
    cdef list categorias_pequenas
    cdef set categorias_set
    cdef object valor_pequeno = 'Otro'
    cdef object valor_desconocido = 'Desconocido'

    for col in range(n_cols):
        frecuencia = {}

        # Contar la frecuencia de cada categoría
        for row in range(n_rows):
            category = data[row][col]
            if category is not None:
                if category in frecuencia:
                    frecuencia[category] += 1
                else:
                    frecuencia[category] = 1
            else:
                # Contar NaN como una categoría especial
                if 'NaN' in frecuencia:
                    frecuencia['NaN'] += 1
                else:
                    frecuencia['NaN'] = 1

        # Identificar categorías raras (frecuencia < umbral)
        categorias_pequenas = [key for key, value in frecuencia.items() if value < umbral and key != 'NaN']
        categorias_set = set(categorias_pequenas)

        # Reemplazar categorías pequeñas y NaN
        for row in range(n_rows):
            category = data[row][col]
            if category in categorias_set:
                data[row][col] = valor_pequeno
            elif category is None:
                data[row][col] = valor_desconocido

    return data

def custom_one_hot_encoder_cython(list data, int delimiter=124):
    """
    Codifica los datos de lista en formato one-hot.
    
    Parámetros:
    - data (list of lists): Matriz de datos.
    - delimiter (int): Delimitador para las categorías (por defecto es '|' ASCII 124).
    
    Retorna:
    - unique_categories (list): Lista de categorías únicas.
    - binary_matrix (list of lists): Matriz binaria one-hot.
    """
    cdef int i, j
    cdef int n_rows = len(data)
    cdef set unique_set = set()
    cdef list row
    cdef object category

    # Extraer las categorías únicas
    for i in range(n_rows):
        row = data[i]
        for j in range(len(row)):
            category = row[j]
            if category is not None:
                unique_set.add(str(category))  # Convertir todo a string
            else:
                unique_set.add('Desconocido')

    # Convertir a lista y ordenar
    unique_categories = sorted(list(unique_set))  # sorted después de convertir a str
    cdef int n_categories = len(unique_categories)

    cdef dict category_to_index = {}
    for i in range(n_categories):
        category_to_index[unique_categories[i]] = i

    # Crear la matriz binaria
    cdef list binary_matrix = []
    cdef list binary_row
    cdef int index

    for i in range(n_rows):
        binary_row = [0] * n_categories
        row = data[i]
        for j in range(len(row)):
            category = str(row[j]) if row[j] is not None else 'Desconocido'  # Convertir a string
            if category in category_to_index:
                index = category_to_index[category]
                binary_row[index] = 1
        binary_matrix.append(binary_row)

    return unique_categories, binary_matrix

def boolean_features_ohe_cython(list list_data, list unique_values):
    """
    Optimiza el one-hot encoding de listas de datos con Cython.
    
    Parámetros:
    - list_data (list of lists): Listas de datos de las columnas a procesar.
    - unique_values (list): Lista de valores únicos para el one-hot encoding.

    Retorna:
    - ohe_result (list of lists): Lista de listas con el resultado del one-hot encoding.
    """
    cdef int num_rows = len(list_data[0])  # Número de filas
    cdef int num_columns = len(list_data)  # Número de columnas
    
    # Crear una lista de resultados con ceros
    cdef list ohe_result = [[0] * len(unique_values) for _ in range(num_rows)]
    cdef int i, j, k
    
    # Iterar sobre las filas
    for i in range(num_rows):
        for j in range(num_columns):
            current_value = list_data[j][i]
            if current_value in unique_values:
                # Marcar la posición correspondiente como 1
                k = unique_values.index(current_value)
                ohe_result[i][k] = 1
    
    return ohe_result

def verificar_festividades_cython(list auction_times):
    """
    Verifica si una lista de fechas de subastas está cerca de una festividad importante sin utilizar pandas.
    
    Parámetros:
    - auction_times (list): Lista de fechas de subasta como tuplas de formato (año, mes, día).

    Retorna:
    - (list): Lista de 1 o 0 indicando si la fecha está cerca de una festividad importante.
    """
    cdef int n = len(auction_times)
    cdef int i, year, month, day
    cdef list resultado = [0] * n

    # Definir las festividades globales
    cdef list festividades = [
        (12, 25),  # Navidad
        (12, 31),  # Fin de Año
        (1, 6),    # Reyes Magos
        (2, 14),   # San Valentín
        (11, 29),  # Black Friday
        (12, 2),   # Cyber Monday
        (10, 31),  # Halloween
        (4, 1),    # Pascua ajustada
        (5, 12),   # Día de la Madre
        (6, 16),   # Día del Padre
        (8, 11)    # Día del Niño
    ]

    for i in range(n):
        year, month, day = auction_times[i]

        for fest_month, fest_day in festividades:
            # Verificar si la subasta está dentro de los 10 días previos a la festividad
            if month == fest_month and (fest_day - 10) <= day <= fest_day:
                resultado[i] = 1
                break  # Salimos del loop al encontrar una festividad cercana
            # Caso especial para Reyes Magos (enero después de diciembre)
            elif month == 12 and fest_month == 1 and day >= 22:
                resultado[i] = 1
                break

    return resultado

def agrupar_edades_cython(list edades):
    """
    Agrupa las edades en rangos numéricos para mejorar la predicción usando Cython.

    Parámetros:
    - edades (list): Lista de edades.

    Retorna:
    - age_groups (list): Lista con los rangos de edad.
    """
    cdef int n = len(edades)
    cdef int i
    cdef float edad
    cdef list age_groups = [0] * n
    
    for i in range(n):
        edad = edades[i]
        
        if edad < 0 or edad > 100:
            age_groups[i] = 0  # Atípico
        elif 0 <= edad <= 18:
            age_groups[i] = 1  # Niños/Adolescentes
        elif 19 <= edad <= 29:
            age_groups[i] = 2  # Jóvenes Adultos
        elif 30 <= edad <= 45:
            age_groups[i] = 3  # Adultos
        elif 46 <= edad <= 60:
            age_groups[i] = 4  # Adultos Mayores
        else:
            age_groups[i] = 5  # Personas Mayores (61-100)
    
    return age_groups

def expand_action_list_0_cython(list action_list_0, list existing_columns, list current_matrix):
    """
    Expande la columna 'action_list_0' en valores únicos y marca con 1 las columnas existentes o las crea si es necesario.
    
    Parámetros:
    - action_list_0 (list): Lista con los valores de 'action_list_0'.
    - existing_columns (list): Lista de nombres de columnas que ya existen en el DataFrame.
    - current_matrix (list of lists): Matriz actual que representa las columnas del DataFrame.

    Retorna:
    - updated_matrix (list of lists): Matriz actualizada con las columnas de valores únicos de 'action_list_0'.
    """
    cdef int num_rows = len(action_list_0)
    cdef int num_columns = len(existing_columns)
    cdef int i, j
    cdef str value

    # Iterar sobre las filas de 'action_list_0'
    for i in range(num_rows):
        value = action_list_0[i]
        
        if value in existing_columns:
            # Buscar el índice de la columna correspondiente
            col_idx = existing_columns.index(value)
            current_matrix[i][col_idx] = 1  # Marcar con 1 si no está marcado
        else:
            # Agregar la nueva columna si no existe
            existing_columns.append(value)
            # Expandir la matriz con una nueva columna de 0s
            for row in current_matrix:
                row.append(0)
            # Marcar la nueva columna en la fila correspondiente
            current_matrix[i][len(existing_columns) - 1] = 1
    
    return current_matrix
# tools.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libc.time cimport time, strftime

cpdef list agrupar_categorias_cython(list categorical_features, list data, int umbral=100):
    """
    Agrupa categorías raras en 'Otro' usando Cython para mejorar el rendimiento.
    No modifica los valores desconocidos (None).
    Si umbral <= 0, retorna los datos sin modificaciones.
    
    Parámetros:
    - categorical_features (list): Lista de nombres de columnas categóricas a procesar.
    - data (list of lists): Conjunto de datos donde cada sublista representa una fila.
    - umbral (int): Umbral de frecuencia para considerar una categoría como rara.
    
    Retorna:
    - data (list of lists): Conjunto de datos modificado.
    """
    # Si el umbral no es mayor a 0, retornar los datos sin modificaciones
    if umbral <= 0:
        return data

    cdef int col, row
    cdef int n_cols = len(categorical_features)
    cdef int n_rows = len(data)
    cdef object category
    cdef dict frecuencia
    cdef list categorias_pequenas
    cdef set categorias_set
    cdef object valor_pequeno = 'Otro'

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
                # No contar 'None' para evitar reemplazo
                pass

        # Identificar categorías raras (frecuencia < umbral)
        categorias_pequenas = [key for key, value in frecuencia.items() if value < umbral]
        categorias_set = set(categorias_pequenas)

        # Reemplazar únicamente categorías pequeñas
        for row in range(n_rows):
            category = data[row][col]
            if category in categorias_set:
                data[row][col] = valor_pequeno
            # No modificar 'None'

    return data


cpdef tuple custom_one_hot_encoder_cython(list data, int delimiter=124):
    """
    Codifica los datos de lista en formato one-hot.
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

cpdef list agrupar_edades_cython(list edades):
    """
    Agrupa las edades en rangos numéricos para mejorar la predicción usando Cython.
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

cpdef list expand_action_list_0_cython(list action_list_0, list existing_columns, list current_matrix):
    """
    Expande la columna 'action_list_0' en valores únicos y marca con 1 las columnas existentes o las crea si es necesario.
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

cpdef list boolean_features_ohe_cython(list list_data, list unique_values):
    """
    Función de codificación one-hot personalizada que maneja valores de tipo string.
    
    Parámetros:
    - list_data (list of lists): Lista de columnas a codificar, donde cada columna es una lista de valores.
    - unique_values (list): Lista de valores únicos para crear las columnas de OHE.
    
    Retorna:
    - ohe_result (list of lists): Lista de filas codificadas en one-hot.
    """
    cdef int num_rows = len(list_data[0])
    cdef int num_unique = len(unique_values)
    cdef list ohe_result = []
    cdef int i, j
    cdef object current_value

    # Crear un diccionario para mapear unique_values a índices para una búsqueda más rápida
    cdef dict unique_map = {}
    for j in range(num_unique):
        unique_map[unique_values[j]] = j

    # Inicializar una lista de listas con ceros
    ohe_result = [ [0] * num_unique for _ in range(num_rows) ]

    # Iterar sobre cada columna y cada fila para establecer los valores de OHE
    for j in range(len(list_data)):
        for i in range(num_rows):
            current_value = list_data[j][i]
            if current_value in unique_map:
                ohe_result[i][unique_map[current_value]] = 1

    return ohe_result

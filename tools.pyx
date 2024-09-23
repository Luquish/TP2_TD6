# tools.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

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
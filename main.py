# Загрузим необходимые библиотеки
import numpy as np
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')
print('1) Ручной ввод\n'
      '2) Задание')
select = input('->')

global m_gl, n_gl
if select == '2':
    #Запишем данные в массивы
    a = np.array([220, 90, 101])
    b = np.array([40, 60, 120, 100])

    D = np.array([[11, 39, 7, 3],
                  [7, 11, 39, 15],
                 [19, 31, 27, 11]])

    m_gl = len(a)
    n_gl = len(b)
else:
    #n = int(input('Введите количество поставщиков(строк): '))
    a = np.array(list(map(int ,input('Введите запасы через пробел: ').split(' '))))

    #m = int(input('Введите количество потребителей(столбцов): '))
    b = np.array(list(map(int ,input('Введите потребности через пробел: ').split(' '))))

    D_list = []
    print('Ввод матрицы поставок')
    for i in range(1, len(a) + 1):
        D_list.append(list(map(int, input('Ввод '+ str(i) +' строки: ').split(' '))))
    D = np.array(D_list)

    m_gl = len(a)
    n_gl = len(b)

# Необходима функция нахождения индексов минимального элемента матрицы
def ij(c_min):
    c = np.inf
    for i in range(c_min.shape[0]):
        for j in range(c_min.shape[1]):
            if (c_min[i, j] != 0) and (c_min[i, j] < c):
                c = c_min[i, j]
                i_, j_ = i, j
    return i_, j_


# Функция минимального элемента
def M_min(a_, b_, c_, print_=False):
    a = np.copy(a_)
    b = np.copy(b_)
    c = np.copy(c_)

    # Проверяем условие замкнутости: если не замкнута - меняем соотвествующие векторы и матрицу трансп. расходов
    if a.sum() > b.sum():
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.vstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)
    x = np.zeros((m, n), dtype=int)  # создаем матрицу для x и заполняем нулями
    x_for_arr = [[-999 for y in range(n)] for x in range(m)]
    x_arr = np.array(x_for_arr)
    #print(x_arr)
    funk = 0
    while True:
        c_min = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                c_min[i, j] = (c[i, j] * min(a[i], b[j]))  # составляем матрицу суммарных расходов
        i, j = ij(c_min)  # определяем индексы минимального элемента составленной матрицы суммарных расходов
        x_ij = int(min(a[i], b[j]))
        x[i, j] = x_ij  # добавляем элемент x_ij в матрицу x
        x_arr[i, j] = x_ij
        funk += int(c_min[i, j])  # добавляем x_ij в итоговую функцию
        a[i] -= x_ij  #
        b[j] -= x_ij  # обновляем векторы a и b
        if print_:
            print('c_min:')
            print(c_min.astype(int))
            print('a: ', a)
            print('b: ', b)
            print()
        if len(c_min[c_min > 0]) == 1:  # повторяем до сходимости метода
            break
    #print(x_arr)
    return x, funk  # возращаем матрицу x и целевую функцию

#x, funk = M_min(a, b, D, print_ = True)
#print('x: ')
#print(x)
#print('Целевая функция: ', funk)


def sev_zap(a_, b_, c_):
    a = np.copy(a_)
    b = np.copy(b_)
    c = np.copy(c_)

    # Проверяем условие замкнутости:
    if a.sum() > b.sum():
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.vstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)
    i = 0
    j = 0
    funk = 0
    x = np.zeros((m, n), dtype=int)
    x_for_arr = [[-999 for y in range(n)] for x in range(m)]
    x_arr = np.array(x_for_arr)
    #print(x_arr)
    while (i < m) and (j < n):  # повторяем цикл до сходимости метода
        x_ij = min(a[i], b[j])  # проверяем минимальность a_i и b_j
        funk += c[i, j] * min(a[i], b[j])  # записываем в итоговую функцию элемент трат
        a[i] -= x_ij  #
        b[j] -= x_ij  # обновляем векторы a и b
        x[i, j] = x_ij  # добавляем элемент x_ij в матрицу x
        x_arr[i, j] = x_ij

        if a[i] > b[j]:  # делаем сдвиги при выполнении условий
            j += 1
        elif a[i] < b[j]:
            i += 1
        elif a[i] == b[j]:
            i += 1
        else:
            i += 1
            j += 1
    #print(x_arr)
    #return x, funk  # возращаем матрицу x и целевую функцию
    return x, x_arr, funk  # возращаем матрицу x, x_arr и целевую функцию


# Для метода потенциалов потребуется матрица дельт
# На вход она получает x - матрицу одного из опорных методов
def delta(a, b, c, x, x_arr):
    # Проверяем условие замкнутости:
    if a.sum() > b.sum():
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.vstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)

    u = np.zeros(m)
    v = np.zeros(n)

    for i in range(m):
        for j in range(n):
            if x_arr[i, j] != -999:  # если элемент матрицы x не равен 0, расчитываем для данных индексов векторы u и v
                if v[j] != 0:
                    u[i] = c[i, j] - v[j]
                else:
                    v[j] = c[i, j] - u[i]
    #print('Потенциалы u ', u)
    #print('Потенциалы v ', v)
    delta = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if x_arr[i, j] == -999:  # если элемент матрицы x равен 0
                delta[i, j] = c[i, j] - (u[i] + v[j])  # расчитываем элемент дельта матрицы
    return delta


# Функция возвращает матрицу системы ограничений
def prepare(a, b):
    m = len(a)
    n = len(b)
    h = np.diag(np.ones(n))
    v = np.zeros((m, n))
    v[0] = 1
    for i in range(1, m):
        h = np.hstack((h, np.diag(np.ones(n))))
        k = np.zeros((m, n))
        k[i] = 1
        v = np.hstack((v, k))
    return np.vstack((h, v)).astype(int), np.hstack((b, a))


# Метод потенциалов
def potenz(x_, x_arr_, delta_, c_):
    x = np.copy(x_)
    x_arr = np.copy(x_arr_)
    delta = np.copy(delta_)
    c = np.copy(c_)

    m_i = 0 # индекс минимальной дельты
    m_j = 0 # индекс минимальной дельты
    min_delta = 9999

    for i in range(0, m_gl): # Нахождение минимальной дельты
        for j in range(0, n_gl):
            if delta[i, j] < min_delta:
                min_delta = delta[i, j]
                m_i = i
                m_j = j

    arr_i = []
    arr_j = []
    for i in range(0, m_gl): # Цикл
        if (x_arr[i, m_j] != -999) and (i != m_i):
            for j in range(0, n_gl):
                if (x_arr[m_i, j] != -999) and (j != m_j) and (x_arr[i, j] != -999):
                    arr_i.append(i) # 1 угловая точка
                    arr_j.append(m_j)

                    arr_i.append(i) # 2 угловая точка
                    arr_j.append(j) # Она же диагональна нашей дельте точке

                    arr_i.append(m_i) # 3 угловая точка
                    arr_j.append(j)
    if len(arr_i) == 0:
        print('Не удалось построить цикл')
        exit()
    out_from_basis = 0
    if x[arr_i[0], arr_j[0]] < x[arr_i[-1], arr_j[-1]]: # Определяем выводимую клетку из базиса
        out_from_basis = x[int(arr_i[0]), int(arr_j[0])]
        x_arr[arr_i[0], arr_j[0]] = -999 # Меняем значение выводимой клетки в базисной таблице
    else:
        out_from_basis = x[int(arr_i[-1]), int(arr_j[-1])]
        x_arr[int(arr_i[-1]), int(arr_j[-1])] = -999  # Меняем значение в базисной таблице


    x[arr_i[0], arr_j[0]] -= out_from_basis  # Вычитаем выводимую клетку
    x[arr_i[-1], arr_j[-1]] -= out_from_basis  # Вычитаем выводимую клетку
    x[m_i, m_j] = out_from_basis # Прибавляем вводимую клетку
    x[arr_i[1], arr_j[1]] += out_from_basis # Прибавляем вводимую клетку

    x_arr[m_i, m_j] = out_from_basis # Меняем значение вводимой клетки в базисной таблице

    return x, x_arr


def checkOptimal(deltaM):
    min_el = 999
    for i in range(0, m_gl): # Цикл
            for j in range(0, n_gl):
                if deltaM[i, j] < min_el:
                    min_el = deltaM[i, j]
    if min_el >= 0:
        return True
    else:
        return False

print()

print('Опорный план по методу\n'
      '1) Метод минимального элемента\n'
      '2) Метод северо-западного угла')
if input('->') == '1':
    x, funk = M_min(a, b, D)
    print('\nМетод минимального элемента \n', x)
    print('Целевая функция: ', funk)
    print()
    print('Дельта матрица для опорного плана: \n', delta(a, b, D, x))
else:
    #x, funk = sev_zap(a, b, D)
    x, x_arr, funk = sev_zap(a, b, D)
    deltaMatrix = delta(a, b, D, x, x_arr)
    print('-'*120)
    print('Метод северо-западного угла \n', x)
    print('Целевая функция: ', funk)
    print('Базисная матрица: \n', x_arr)
    print('Дельта матрица для опорного плана: \n', deltaMatrix)
    print('-' * 120)
    optimal = checkOptimal(deltaMatrix)
    if optimal == True:
        exit()

    for i in range(0, 5):
        x, x_arr = potenz(x, x_arr, deltaMatrix, D)
        function = 0
        print('-' * 120)
        print('Метод потенциалов \n', x)
        for i in range(0, m_gl):
            for j in range(0, n_gl):
                function += x[i, j] * D[i, j]
        print('Ответ: ', function)
        deltaMatrix = delta(a, b, D, x, x_arr)
        print('Базисная матрица: \n', x_arr)
        print('Дельта матрица для метода потенциалов: \n', deltaMatrix)
        print('-' * 120)
        optimal = checkOptimal(deltaMatrix)
        if optimal == True:
            break



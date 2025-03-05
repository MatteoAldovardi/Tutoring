import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.calculus.util import continuous_domain

def analisi_funzione(funzione_str, intervallo=(-10, 10), punti=400):
    """
    Calcola e visualizza dominio, segno, derivata prima e seconda, e grafico di una funzione.

    Args:
        funzione_str (str): La funzione come stringa (es. "x**2 - 4 / (x - 1)").
        intervallo (tuple): Intervallo iniziale di valori x per il grafico (es. (-10, 10)).
        punti (int): Numero di punti per il grafico.
    """
    x = sp.symbols('x')
    funzione = sp.sympify(funzione_str)

    # Calcolo derivate
    derivata_prima = sp.diff(funzione, x)
    derivata_seconda = sp.diff(derivata_prima, x)

    # Determinazione dominio
    dominio = continuous_domain(funzione, x, sp.S.Reals)

    # Studio del segno della funzione
    try:
        segno_funzione = sp.solve_univariate_inequality(funzione > 0, x)
        print(f"Segno funzione: {segno_funzione}")
    except NotImplementedError:
        print("Impossibile determinare il segno della funzione simbolicamente.")

    # Studio del segno della derivata prima
    try:
        segno_derivata_prima = sp.solve_univariate_inequality(derivata_prima > 0, x)
        print(f"Segno derivata prima: {segno_derivata_prima}")
    except NotImplementedError:
        print("Impossibile determinare il segno della derivata prima simbolicamente.")

    # Studio del segno della derivata seconda
    try:
        segno_derivata_seconda = sp.solve_univariate_inequality(derivata_seconda > 0, x)
        print(f"Segno derivata seconda: {segno_derivata_seconda}")
    except NotImplementedError:
        print("Impossibile determinare il segno della derivata seconda simbolicamente.")

    # Stampa dei risultati
    print(f"Funzione: {funzione}")
    print(f"Dominio: {dominio}")
    print(f"Derivata prima: {derivata_prima}")
    print(f"Derivata seconda: {derivata_seconda}")

    # Plot del grafico
    x_valori = np.linspace(intervallo[0], intervallo[1], punti)
    funzione_valori = [funzione.subs(x, val) for val in x_valori]
    derivata_prima_valori = [derivata_prima.subs(x, val) for val in x_valori]
    derivata_seconda_valori = [derivata_seconda.subs(x, val) for val in x_valori]

    # Filtro dei valori complessi
    funzione_valori = [float(val) if sp.im(val) == 0 else np.nan for val in funzione_valori]
    derivata_prima_valori = [float(val) if sp.im(val) == 0 else np.nan for val in derivata_prima_valori]
    derivata_seconda_valori = [float(val) if sp.im(val) == 0 else np.nan for val in derivata_seconda_valori]

    # Adattamento intervallo grafico
    if sp.FiniteSet in type(dominio).__mro__:
        dominio = sp.Interval(min(dominio), max(dominio))
    if sp.Interval in type(dominio).__mro__:
        if dominio.left is not sp.S.NegativeInfinity and dominio.right is not sp.S.Infinity:
            x_valori = np.linspace(float(dominio.left), float(dominio.right), punti)
            funzione_valori = [funzione.subs(x, val) for val in x_valori]
            derivata_prima_valori = [derivata_prima.subs(x, val) for val in x_valori]
            derivata_seconda_valori = [derivata_seconda.subs(x, val) for val in x_valori]

            # Filtro dei valori complessi dopo l'adattamento
            funzione_valori = [float(val) if sp.im(val) == 0 else np.nan for val in funzione_valori]
            derivata_prima_valori = [float(val) if sp.im(val) == 0 else np.nan for val in derivata_prima_valori]
            derivata_seconda_valori = [float(val) if sp.im(val) == 0 else np.nan for val in derivata_seconda_valori]

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(x_valori, funzione_valori, label='f(x)')
    plt.title('Funzione')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x_valori, derivata_prima_valori, label="f'(x)")
    plt.title('Derivata prima')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x_valori, derivata_seconda_valori, label="f''(x)")
    plt.title('Derivata seconda')
    plt.legend()

    plt.tight_layout()
    plt.show()


def limite(funzione_str, variabile_str, valore_str, direzione_str='from_right'):
    """Calcola il limite di una funzione."""
    x = sp.symbols(variabile_str)
    funzione = sp.sympify(funzione_str)
    valore = sp.sympify(valore_str)
    direzione = '+' if direzione_str == 'from_right' else '-'
    return sp.limit(funzione, x, valore, dir=direzione)

def derivata(funzione_str, variabile_str, ordine=1):
    """Calcola la derivata di una funzione e la restituisce come stringa."""
    x = sp.symbols(variabile_str)
    funzione = sp.sympify(funzione_str)
    derivata = sp.diff(funzione, x, ordine)
    return str(derivata)



def asintoti_obliqui(funzione_str, variabile_str):
    """Calcola l'asintoto obliquo di una funzione."""
    """ Input : Stringa della funzione """
    """ Output : Ritorna nell'ordine: 
        Eventuale asintoto a +infty :(coeffiente_angolare, intercetta)
        Eventuale asintoto a -infty :(coeffiente_angolare, intercetta)
    """
    x = sp.symbols(variabile_str)
    funzione = sp.sympify(funzione_str)
    try:
        # Calcolo asintoto per x -> +oo
        m_pos = sp.limit(funzione / x, x, sp.oo)
        q_pos = sp.limit(funzione - m_pos * x, x, sp.oo)

        # Calcolo asintoto per x -> -oo
        m_neg = sp.limit(funzione / x, x, -sp.oo)
        q_neg = sp.limit(funzione - m_neg * x, x, -sp.oo)

        # Verifica esistenza asintoti
        if m_pos != 0 and q_pos != sp.oo:
            asintoto_pos = (m_pos, q_pos)
        else:
            asintoto_pos = None

        if m_neg != 0 and q_neg != -sp.oo:
            asintoto_neg = (m_neg, q_neg)
        else:
            asintoto_neg = None

        return asintoto_pos, asintoto_neg

    except sp.calculus.util.PoleError:
        return None, None



if __name__ == "__main__":
    x = sp.symbols('x')
    # Esempio di utilizzo (opzionale)
    analisi_funzione("log(x)")
    analisi_funzione("exp(-x**2)")
    analisi_funzione("(log(x+1))/(x-2)")
    analisi_funzione("cos(x)")
    print(derivata(str(sp.exp(x)), "x")) 
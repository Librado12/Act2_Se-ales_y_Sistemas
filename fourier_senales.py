"""
Análisis de señales en el dominio del tiempo y de la frecuencia
con la Transformada de Fourier (FFT) usando NumPy y Matplotlib.

Estructura:
  1. Parámetros de muestreo
  2. Definición de señales en el dominio del tiempo
  3. Cálculo de la Transformada de Fourier
  4. Visualización de resultados (tiempo, magnitud y fase)
  5. Verificación de propiedades: linealidad, desplazamiento en el tiempo
     y escalamiento
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. PARÁMETROS DE MUESTREO
# ---------------------------------------------------------------------------
fs = 1000                  # frecuencia de muestreo: 1000 muestras por segundo (Hz)
T = 1.0                    # duración de la señal en segundos
N = int(fs * T)            # número total de muestras
t = np.arange(N) / fs      # vector de tiempo: 0, 0.001, 0.002, ... 0.999


# ---------------------------------------------------------------------------
# 2. SEÑALES EN EL DOMINIO DEL TIEMPO
#    Cada función recibe el tiempo "t" y devuelve el valor de la señal.
# ---------------------------------------------------------------------------
def pulso_rectangular(t, inicio=0.2, fin=0.3):
    """Vale 1 entre 'inicio' y 'fin', y 0 en el resto."""
    return ((t >= inicio) & (t < fin)).astype(float)


def escalon(t, t0=0.5):
    """Función escalón: vale 0 antes de t0 y 1 desde t0."""
    return (t >= t0).astype(float)


def senoidal(t, f=50, A=1.0):
    """Senoidal de frecuencia f (Hz) y amplitud A."""
    return A * np.sin(2 * np.pi * f * t)


# ---------------------------------------------------------------------------
# 3. TRANSFORMADA DE FOURIER
# ---------------------------------------------------------------------------
def calcular_fft(x, fs):
    """
    Devuelve: frecuencias, X (complejo), magnitud y fase.
    El espectro se centra en 0 Hz con fftshift para verlo de -fs/2 a +fs/2.
    """
    n = len(x)
    X = np.fft.fft(x)                      # transformada (números complejos)
    f = np.fft.fftfreq(n, d=1 / fs)        # frecuencia (Hz) de cada punto
    X = np.fft.fftshift(X)                 # reordenar: negativas | 0 | positivas
    f = np.fft.fftshift(f)

    magnitud = np.abs(X) / n               # dividir entre n normaliza la amplitud
    fase = np.angle(X)                     # fase en radianes (entre -pi y pi)

    # Donde la magnitud es casi 0, la fase no tiene sentido (es ruido numérico)
    fase[magnitud < 1e-3] = 0
    return f, X, magnitud, fase


# ---------------------------------------------------------------------------
# 4. VISUALIZACIÓN
# ---------------------------------------------------------------------------
def graficar(x, nombre, archivo, fmax=150):
    f, X, mag, fase = calcular_fft(x, fs)

    fig, ejes = plt.subplots(3, 1, figsize=(9, 9))

    ejes[0].plot(t, x)
    ejes[0].set_title(f"{nombre} - Dominio del tiempo")
    ejes[0].set_xlabel("Tiempo (s)")
    ejes[0].set_ylabel("Amplitud")

    ejes[1].plot(f, mag)
    ejes[1].set_title("Espectro de magnitud")
    ejes[1].set_xlabel("Frecuencia (Hz)")
    ejes[1].set_ylabel("|X(f)|")
    ejes[1].set_xlim(-fmax, fmax)

    ejes[2].plot(f, fase)
    ejes[2].set_title("Espectro de fase")
    ejes[2].set_xlabel("Frecuencia (Hz)")
    ejes[2].set_ylabel("Fase (rad)")
    ejes[2].set_xlim(-fmax, fmax)

    for ax in ejes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"resultados/{archivo}.png", dpi=150)


x_pulso = pulso_rectangular(t)
x_escalon = escalon(t)
x_seno = senoidal(t, f=50)

graficar(x_pulso, "Pulso rectangular", "pulso_rectangular")
graficar(x_escalon, "Función escalón", "escalon")
graficar(x_seno, "Senoidal de 50 Hz", "senoidal")


# ---------------------------------------------------------------------------
# 5. PROPIEDADES DE LA TRANSFORMADA DE FOURIER
# ---------------------------------------------------------------------------

# --- 5.1 LINEALIDAD:  FFT(a*x1 + b*x2) = a*FFT(x1) + b*FFT(x2) -------------
a, b = 2, 3
lado_izq = np.fft.fft(a * x_pulso + b * x_seno)
lado_der = a * np.fft.fft(x_pulso) + b * np.fft.fft(x_seno)
print("Linealidad se cumple:", np.allclose(lado_izq, lado_der))

# --- 5.2 DESPLAZAMIENTO EN EL TIEMPO ---------------------------------------
# Si x(t) se retrasa n0 muestras, la magnitud NO cambia, solo la fase:
#   FFT(x[n - n0]) = FFT(x[n]) * exp(-j*2*pi*k*n0/N)
n0 = 100                                   # 100 muestras = 0.1 s
x_desplazada = np.roll(x_pulso, n0)        # desplazamiento circular
k = np.arange(N)
X_original = np.fft.fft(x_pulso)
X_desplazada = np.fft.fft(x_desplazada)
X_teorica = X_original * np.exp(-1j * 2 * np.pi * k * n0 / N)

print("Magnitud igual tras desplazar:",
      np.allclose(np.abs(X_original), np.abs(X_desplazada)))
print("Desplazamiento coincide con la teoría:",
      np.allclose(X_desplazada, X_teorica))

f, _, mag1, fase1 = calcular_fft(x_pulso, fs)
_, _, mag2, fase2 = calcular_fft(x_desplazada, fs)

fig, ejes = plt.subplots(3, 1, figsize=(9, 9))
ejes[0].plot(t, x_pulso, label="Original")
ejes[0].plot(t, x_desplazada, "--", label="Desplazada 0.1 s")
ejes[0].set_title("Desplazamiento en el tiempo")
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].legend()
ejes[1].plot(f, mag1, label="Original")
ejes[1].plot(f, mag2, "--", label="Desplazada")
ejes[1].set_title("Magnitud (idéntica)")
ejes[1].set_xlim(-60, 60)
ejes[1].set_xlabel("Frecuencia (Hz)")
ejes[1].legend()
ejes[2].plot(f, fase1, label="Original")
ejes[2].plot(f, fase2, label="Desplazada")
ejes[2].set_title("Fase (cambia)")
ejes[2].set_xlim(-60, 60)
ejes[2].set_xlabel("Frecuencia (Hz)")
ejes[2].legend()
for ax in ejes:
    ax.grid(True)
fig.tight_layout()
fig.savefig("resultados/propiedad_desplazamiento.png", dpi=150)

# --- 5.3 ESCALAMIENTO ------------------------------------------------------
# Comprimir en el tiempo (x(2t)) ensancha el espectro, y viceversa.

# (a) Con la senoidal: x(2t) duplica la frecuencia del pico (50 Hz -> 100 Hz)
def frecuencia_pico(x):
    fr, _, m, _ = calcular_fft(x, fs)
    return fr[np.argmax(m)] if fr[np.argmax(m)] > 0 else abs(fr[np.argmax(m)])

x_seno_comprimida = senoidal(2 * t, f=50)   # equivale a senoidal(t, f=100)
print("Pico original:", abs(frecuencia_pico(x_seno)), "Hz")
print("Pico con x(2t):", abs(frecuencia_pico(x_seno_comprimida)), "Hz")

# (b) Con el pulso: un pulso más angosto tiene un espectro más ancho
x_pulso_comprimido = pulso_rectangular(2 * t)   # dura la mitad

f, _, mag_a, _ = calcular_fft(x_pulso, fs)
_, _, mag_b, _ = calcular_fft(x_pulso_comprimido, fs)

fig, ejes = plt.subplots(2, 1, figsize=(9, 7))
ejes[0].plot(t, x_pulso, label="x(t): 0.1 s de ancho")
ejes[0].plot(t, x_pulso_comprimido, label="x(2t): 0.05 s de ancho")
ejes[0].set_title("Escalamiento: señales en el tiempo")
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].legend()
ejes[1].plot(f, mag_a, label="Espectro de x(t)")
ejes[1].plot(f, mag_b, label="Espectro de x(2t)")
ejes[1].set_title("Pulso angosto en tiempo = espectro ancho en frecuencia")
ejes[1].set_xlabel("Frecuencia (Hz)")
ejes[1].set_xlim(-80, 80)
ejes[1].legend()
for ax in ejes:
    ax.grid(True)
fig.tight_layout()
fig.savefig("resultados/propiedad_escalamiento.png", dpi=150)

plt.show()

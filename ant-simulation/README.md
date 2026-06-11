# 🐜 Simulación de colonia de hormigas

Simulación 2D de una colonia de hormigas que busca comida usando **solo
feromonas**, sin conocimiento global del mapa. El comportamiento colectivo
(rastros, caminos óptimos, abandono de fuentes agotadas) emerge de reglas
locales muy simples.

## ▶️ Verla en vivo

**https://miguelrepo.github.io/ant-colony-simulation/**

Funciona en cualquier navegador moderno, incluido el del celular.

## Cómo ejecutarla en local

Abre `index.html` en cualquier navegador. No necesita servidor ni
dependencias. Si lo prefieres:

```bash
python3 -m http.server 8000
# → http://localhost:8000
```

## Cómo funciona

Las hormigas son **ciegas**: solo perciben con 3 antenas cortas (izquierda,
centro, derecha) que detectan dos feromonas depositadas en el suelo:

| Feromona | Color | Quién la deposita | Quién la sigue |
|----------|-------|-------------------|----------------|
| **Casa** | Azul | Toda hormiga que sale a explorar | Las que vuelven (con o sin comida) |
| **Comida** | Roja | Las que vuelven cargadas con comida | Las exploradoras |

Reglas clave:

- **Evaporación**: ambas feromonas se desvanecen con el tiempo. Un camino
  solo sobrevive si otras hormigas siguen pasando y lo refuerzan.
- **Reserva limitada**: cada hormiga sale con una reserva de feromona que se
  agota gradualmente. Los rastros cerca del origen (nido o comida) son más
  intensos, lo que crea un gradiente que las demás pueden remontar.
- **Caminos cortos ganan**: las rutas cortas se recorren más veces por unidad
  de tiempo, se refuerzan más y acaban dominando. Nadie calcula la ruta
  óptima: emerge sola.
- **Rendición**: si una exploradora pasa demasiado tiempo sin encontrar nada,
  vuelve al nido siguiendo la feromona de casa y vuelve a intentarlo.
- **Comida finita**: cada fuente se agota unidad a unidad. Cuando se acaba,
  su rastro se evapora y la colonia la abandona de forma natural.

## Controles

- **Hormigas**: número de hormigas (se aplica con "Nuevo mapa").
- **Velocidad**: pasos de simulación por fotograma (1×–5×).
- **Evaporación**: persistencia de las feromonas (más bajo = se borran antes).
- **Checkboxes**: mostrar/ocultar cada capa de feromona.
- **Nuevo mapa**: genera un mapa aleatorio con obstáculos y fuentes nuevas.

El mapa se genera proceduralmente (rocas y muros alargados) y se valida con
un *flood fill* para garantizar que toda la comida sea alcanzable desde el
nido.

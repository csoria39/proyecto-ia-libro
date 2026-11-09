# Plantillas para lectores de prueba — *De la Neurona al LLM*

Edición revisada: v18, 377 páginas, ISBN 9798193023975. Cita siempre la **página del PDF** y, si puedes, la sección
(por ejemplo, «pág. 154, 13.4.2»). Así cada comentario se puede localizar sin dudas.

## Plantilla 1 — Lector del público objetivo

*Sabe programar un poco y nunca ha tocado la IA.*

Instrucciones: no hace falta que termines el libro para entregar esto. Responde capítulo a capítulo, a medida que
avanzas, con el capítulo fresco en la cabeza. No esperes al final para intentar recordarlo todo junto. Si dejas un
capítulo a medias, dilo: también es un dato útil.


### Antes de empezar

**A.** ¿Qué sabes de programación? (curso o asignatura, años, lenguajes) ¿Y de matemáticas: derivadas, matrices?

**B.** ¿Qué ruta del prólogo vas a seguir: la del curioso (sin programar), la del programador (de la neurona a GPT-2)
   o la completa?

**C.** Si vas a ejecutar código: sistema operativo (Windows, macOS o Linux), versión de Python y si seguiste el
   Apéndice E o la guía del entorno de desarrollo del repositorio. ¿Cuánto tardaste en tenerlo todo funcionando?


### Por cada capítulo

Puedes contestar en esta tabla y dejar los comentarios largos debajo, numerados por capítulo.

| Cap. | Tiempo (h) | Esfuerzo 1-5 | ¿Ejecutaste el código? | ¿Intentaste los ejercicios antes de ver la solución? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| 9 | | | | |
| 10 | | | | |
| 11 | | | | |
| 12 | | | | |
| 13 | | | | |
| 14 | | | | |
| 15 | | | | |
| 16 | | | | |
| 17 | | | | |
| 18 | | | | |
| 19 | | | | |
| 20 | | | | |
| Apéndices | | | | |

1. Del 1 al 5, ¿cuánto esfuerzo te costó seguirlo? (1 = muy fácil; 5 = tuve que releer varias veces o buscar
   ayuda externa)

2. ¿Hubo algún punto concreto en el que te perdiste? Cita la página y la frase, el párrafo o el ejemplo exacto.
   «No lo entendí» es útil, pero «no entendí por qué se resta el máximo antes de la exponencial en el softmax» es
   mucho más útil.

3. Si ejecutaste el código: ¿la salida coincidió con la que imprime el libro? Si algo falló o dio otros números,
   copia el mensaje de error o la línea distinta. Si no lo ejecutaste, ¿por qué (falta de tiempo, no sabías cómo,
   no te pareció necesario, tu ruta no lo pide)?

4. Los ejercicios: ¿te sirvieron para comprobar si de verdad habías entendido, o los resolviste «copiando el
   patrón» sin estar seguro de por qué funciona?

5. ¿Hubo algo que el libro daba por sabido y tú no lo sabías?

6. Los diagramas y figuras del capítulo, ¿te ayudaron, te sobraron o no los entendiste?


### Al terminar el libro (o la parte que hayas leído)

7. ¿En qué capítulo notaste el salto de dificultad más brusco? ¿Coincide con el nivel que marca el índice con
   asteriscos (* introductorio, ** intermedio, *** avanzado), o te pareció más difícil o más fácil de lo anunciado?

8. Si tuvieras que señalar UN capítulo para eliminar o resumir porque aportó menos que los demás, ¿cuál sería?

9. Si tuvieras que señalar UN capítulo que necesita más ejemplos o más explicación, ¿cuál sería?

10. Explícale a un amigo, sin los términos técnicos del libro:
    a) qué es el Nano-LLM que construiste en el Capítulo 13 y por qué escribe como escribe;
    b) qué pasa en el Capítulo 16 cuando ese mismo código carga los pesos reales de GPT-2, y por qué el resultado
       es tan distinto.

11. ¿Qué fue lo más útil del libro? ¿Y lo que más te frustró?

12. ¿Volverías a elegir este libro, sabiendo lo que sabes ahora, o buscarías otra cosa? ¿Por qué?

13. Erratas: cualquier error de texto, de numeración o de referencia que hayas visto, con su página.


## Plantilla 2 — Lector con formación técnica en ML/IA

*No ha visto el libro antes.*

Instrucciones: el objetivo NO es comprobar si el código funciona. Eso ya está verificado: los 117 listados se han
ejecutado desde una copia limpia, con 118 de 118 ejecuciones correctas (repositorio,
verificaciones/Ejecucion_completa_programas.md). Lo que se busca es tu criterio de experto: ¿dónde simplifica el
libro hasta el punto de que un principiante se llevaría una idea equivocada, y dónde dice más de lo que demuestra?

Para cada hallazgo indica la página y clasifícalo:
- Error: es falso.
- Engañoso: es cierto en el caso del libro, pero el lector generalizará mal.
- Impreciso: mejorable, pero no induce a error.
- Estilo o terminología.
Si puedes, propón la redacción correcta. Un hallazgo con la frase alternativa vale más que diez sin ella.


### Por cada capítulo

1. ¿Alguna simplificación cruza la línea de «simplificado para enseñar» a «técnicamente engañoso»? Cita el
   párrafo exacto.

2. ¿Alguna afirmación te parece incorrecta o desactualizada, aunque esté simplificada a propósito?

3. ¿La terminología es la estándar en la literatura y la industria, o el libro usa términos propios o traducciones
   poco habituales que confundirían a alguien que luego lea documentación o artículos en inglés?

4. Las cifras de los experimentos (semillas, líneas base, conjuntos de validación): ¿el diseño respalda la
   conclusión que saca el texto, o habría que matizarla?

5. Si tuvieras que explicar este concepto a un principiante, ¿qué cambiarías del enfoque del libro?


### Preguntas globales (al terminar)

6. De lo que no cubre, ¿qué echas en falta que se mencione al menos brevemente, aunque no se desarrolle?

7. ¿Algún capítulo se queda corto para el nivel que anuncia (*, ** o ***), en el sentido de que el lector podría
   creer que domina el tema cuando solo ha visto la superficie?

8. ¿Las conclusiones van más allá de lo que se mide? Fíjate sobre todo en:
   - el Capítulo 15, sobre lo que el Nano-LLM memoriza y lo que generaliza;
   - el Capítulo 16, que atribuye la diferencia con GPT-2 al preentrenamiento;
   - el Capítulo 18, sobre evaluación y benchmarks.
   ¿El libro es honesto sobre sus propias limitaciones?

9. El código usa NumPy puro y el Apéndice D repite los programas con PyTorch: ¿la transición a herramientas
   reales está bien resuelta, o el lector saldrá con hábitos que tendrá que desaprender?

10. Las referencias del Apéndice C y las afirmaciones sobre el estado actual del campo (modelos, regulación,
    herramientas), ¿siguen siendo correctas a la fecha de tu lectura?

11. Si un colega técnico te preguntara «¿recomiendas este libro?», ¿qué le dirías y con qué matiz?

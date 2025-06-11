### `main.py` 
Instancia os objetos de cada classe, executa uma série de rotinas de validação para cada módulo e roda o loop principal de simulação. O loop é discretizado no tempo (com passo `dt`) e integra as equações de estado do robô para simular sua resposta a um controlador de trajetória. Ao final, gera gráficos para análise de desempenho.

### `Robot.py` 
Classe que encapsula todos os parâmetros estruturais do manipulador. Atua como um container de dados que descreve o modelo físico do robô.
* **Parâmetros Cinemáticos**: Geometria do robô definida pelos parâmetros de Denavit-Hartenberg (DH).
* **Parâmetros Dinâmicos**: Propriedades inerciais, incluindo a massa de cada elo, o vetor do centro de massa em relação ao frame do elo e o tensor de inércia de cada elo.
* **Parâmetros de Atuadores**: Constantes do motor e relação de transmissão.

### `Kinematics.py`
Implementa os cálculos de mapeamento entre o espaço de juntas e o espaço Cartesiano.
* **Cinemática Direta (FK)**: Calcula a matriz de transformação homogênea do efetuador final ($T_{0}^{n}$) através da composição das matrizes de transformação de cada elo ($A_i$).
* **Cinemática Inversa (IK)**: Resolve o problema da cinemática inversa numericamente. Utiliza um algoritmo de otimização (`scipy.minimize` com SLSQP) para encontrar os ângulos de junta ($q$) que minimizam uma função de custo baseada nos erros de posição e orientação entre o efetuador e o alvo.
* **Jacobiana Geométrica**: Calcula a matriz Jacobiana ($J$), que estabelece a relação linear entre as velocidades das juntas ($\dot{q}$) e as velocidades linear e angular do efetuador final ($v, \omega$).

### `Dynamics.py`
Implementa o modelo dinâmico do manipulador, permitindo a análise de forças e torques.
* **Dinâmica Direta**: Calcula as acelerações das juntas ($\ddot{q}$) resultantes de um vetor de torques aplicados ($\tau$). O método resolve a equação da dinâmica $\ddot{q} = M(q)^{-1} (\tau - C(q, \dot{q}) - G(q))$. As matrizes de Massa **M(q)**, o vetor de Coriolis **C(q, q̇)** e o vetor de Gravidade **G(q)** são computados eficientemente através de múltiplas chamadas ao algoritmo de dinâmica inversa.
* **Dinâmica Inversa**: Calcula os torques de junta ($\tau$) necessários para gerar um determinado movimento ($q, \dot{q}, \ddot{q}$). A implementação utiliza o **Algoritmo Recursivo de Newton-Euler (RNEA)**.

### `Control.py`
Implementa um sistema de controle em malha fechada no espaço de juntas.
* **Controlador PID**: Para cada junta, um controlador **Proporcional-Integral-Derivativo (PID)** independente é utilizado. O torque de comando ($\tau_{cmd}$) é a soma de três componentes:
    * **Proporcional (P)**: Reage ao erro de posição atual ($e = q_d - q$).
    * **Integral (I)**: Acumula os erros de posição passados. Este termo é essencial para eliminar erros residuais em regime permanente.
    * **Derivativo (D)**: Responde à taxa de variação do erro ($\dot{e} = \dot{q}_d - \dot{q}$), atuando para amortecer a resposta do sistema e evitar oscilações.
* **Sintonia de Ganhos**: Os ganhos do controlador ($K_p, K_i, K_d$) são calculados com base em um modelo dinâmico SISO (Single-Input Single-Output) simplificado para cada junta.

import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from core.base import BGElement, BGPort, BGBond, ElementType

import matplotlib.pyplot as plt

class BondGraphSimulator:
    def __init__(self, model, state_space_builder):
        """
        model — объект модели BondGraph
        state_space_builder — объект StateSpaceBuilder, где уже построены A, B, C, D
        """
        self.model = model
        self.ss = state_space_builder

        # Собираем параметры, которые нужны для симуляции
        self.param_syms = self._collect_parameter_symbols()
        self.param_names = [str(p) for p in self.param_syms]

    def _collect_parameter_symbols(self):
        """Собрать все символы-параметры из модели"""
        param_types = {
            ElementType.CAPACITOR, ElementType.INDUCTOR, ElementType.RESISTOR,
            ElementType.SOURCE_EFFORT, ElementType.SOURCE_FLOW,
            ElementType.TRANSFORMER, ElementType.GYRATOR
        }
        params = set()
        for elem in self.model.elements:
            if elem.type in param_types and hasattr(elem, "parameter") and elem.parameter:
                params.add(sp.Symbol(str(elem.parameter)))
        return sorted(list(params), key=lambda s: str(s))

    def list_required_parameters(self):
        """Вернуть список имен всех параметров, которые нужны для симуляции"""
        return self.param_names
    
    def print_simulation_requirements(self):
        """
        User Instructions:
        - What parameters, state variables, and inputs are needed for the simulation,
        - Example values for an RLC circuit.
        """
        print("=== Required Data for Numerical Simulation ===")
        print("System parameters (provide param_values: dict[str, float]):")
        print("   ", ", ".join(self.param_names))
        print("   Example: param_values = {'R1': 10.0, 'C2': 0.001, 'L3': 0.1}")
        print()
        print("State variables (provide x0: list[float], in the same order):")
        print("   ", ", ".join(str(x) for x in self.ss.x_vars))
        print("   Example: x0 = [0.0, 0.0]   # (e.g., q2=0, p3=0)")
        print()
        print("Input variables (provide u_func: Callable[[float], list[float]], in the same order):")
        print("   ", ", ".join(str(u) for u in self.ss.u_vars))
        print("   Example: u_func = lambda t: [1.0]   # if a single input, e.g., SE0=1 always")
        print("           u_func = lambda t: [np.sin(t)]  # sinusoidal input")
        print("-------------------------------------------------------")
        print("Full function call:")
        print("  result = sim.simulate(param_values, t_span=(0, 5), x0=x0, u_func=u_func)")
        print("  t, y = result['t'], result['y']")

        
    def simulate(
        self,
        initial_state: list[float],
        input_sequence: list[list[float]],
        time_steps: list[float],
        sampling_period: float,
        parameter_values: dict[str, float]
    ):
        """
        initial_state: list[float] — initial values of the states, length = number of states  
        input_sequence: list[list[float]] — list of input vectors for each time step (NxM, where N = steps, M = inputs)  
        time_steps: list[float] — list of time points (or np.arange(0, T, Ts))  
        sampling_period: float — sampling period (in seconds)  
        parameter_values: dict[str, float] — system parameter values  
        """
        A, B, C, D = self.ss.A, self.ss.B, self.ss.C, self.ss.D
        x_vars = self.ss.x_vars
        u_vars = self.ss.u_vars

        # Проверка размеров
        num_steps = len(time_steps)
        n_x = len(x_vars)
        n_u = len(u_vars)
        if len(initial_state) != n_x:
            raise ValueError(f"The size of initial_state must be {n_x}")
        if len(input_sequence) != num_steps:
            raise ValueError(f"input_sequence must have the length of {num_steps}")
        for u in input_sequence:
            if len(u) != n_u:
                raise ValueError(f"Each input vector must have the length of {n_u}")

        # Подстановка параметров
        subs = {sp.Symbol(name): val for name, val in parameter_values.items()}
        A_num = np.array(A.subs(subs)).astype(float)
        B_num = np.array(B.subs(subs)).astype(float)
        C_num = np.array(C.subs(subs)).astype(float)
        D_num = np.array(D.subs(subs)).astype(float)

        X = np.zeros((num_steps, n_x))
        Y = np.zeros((num_steps, C_num.shape[0]))
        X[0] = initial_state

        for k in range(num_steps - 1):
            u = np.array(input_sequence[k])
            # Эйлеровский шаг интегрирования (или RK4, если хочется точнее)
            dx = A_num @ X[k] + B_num @ u
            X[k + 1] = X[k] + sampling_period * dx

        # Вычисляем выход
        for k in range(num_steps):
            u = np.array(input_sequence[k])
            Y[k] = C_num @ X[k] + D_num @ u

        return {
            "t": np.array(time_steps),
            "x": X,
            "u": np.array(input_sequence),
            "y": Y
        }


    def plot_simulation_result(self, result, state_names=None, output_names=None, show_states=True, show_outputs=True):
        """
        result: словарь из simulate (t, x, y)
        state_names: список строк для переменных состояния (если None — индексы)
        output_names: список строк для выходов (если None — индексы)
        """
        t = result['t']
        fig, ax = plt.subplots(figsize=(12, 5))
        handles = []

        # --- Состояния ---
        if show_states:
            x = result['x']
            n_states = x.shape[1]
            state_names = state_names or [f"x{i+1}" for i in range(n_states)]
            for i in range(n_states):
                line, = ax.plot(t, x[:, i], label=f"State: {state_names[i]}")
                handles.append(line)

        # --- Выходы ---
        if show_outputs and 'y' in result:
            y = result['y']
            n_outputs = y.shape[1] if len(y.shape) > 1 else 1
            output_names = output_names or [f"y{i+1}" for i in range(n_outputs)]
            for i in range(n_outputs):
                line, = ax.plot(t, y[:, i], '--', label=f"Output: {output_names[i]}")
                handles.append(line)

        ax.set_xlabel("Time, s")
        ax.set_ylabel("Value")
        ax.set_title("Simulation result")
        leg = ax.legend(loc='best', fancybox=True, shadow=True)
        leg_lines = leg.get_lines()

        # --- Интерактивность: клик по легенде включает/выключает линии ---
        def on_legend_pick(event):
            legline = event.artist
            origline = handles[leg_lines.index(legline)]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            legline.set_alpha(1.0 if vis else 0.2)
            fig.canvas.draw()

        for legline in leg_lines:
            legline.set_picker(True)

        fig.canvas.mpl_connect('pick_event', on_legend_pick)
        plt.show()

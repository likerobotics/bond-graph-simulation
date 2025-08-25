# equations/cauchy_form.py
# equations/cauchy_form.py

import sympy as sp
from equations.generator import EquationGenerator
from core.base import ElementType

from typing import List, Optional


class CauchyFormGenerator:
    """
    Builds cauchy-form equations (dot(q), dot(p)) from the bond graph model,
    then recursively eliminates eX, fY by substituting them with expressions
    in terms of final variables (q, p, R, C, I, SE, SF, etc.).
    """

    def __init__(self, model, debug=False):
        self.model = model
        self.debug = debug

        # Full list of 'effort-flow' equations from EquationGenerator
        self.global_equations = []
        # The final cauchy eq => dot(q) or dot(p) = ...
        self.cauchy_equations = []

        # Output eq
        self.output_eqs = []

        # Set of final variables (states, inputs, parameters)
        self.final_vars = list()

        self.state_vars = list()
        self.input_vars = list()
        self.output_vars = list()
        # 

    def add_state_derivative_equations(self, equations):
        """
        Дополняет уравнения связи производных переменных состояния (q, p) с bond-ами.
        Возвращает расширенный список уравнений.
        """
        new_eqs = list(equations)
        # Для C (q): dq/dt = f_bond
        for element in self.model.elements:
            if element.type == ElementType.CAPACITOR:
                q_sym = sp.Symbol(f"q{element.id}")
                dq_sym = sp.Symbol(f"dotq{element.id}")  # обозначение производной
                # Найди bond, соединённый с этим элементом
                for bond in self.model.bonds:
                    if bond.from_port in element.ports or bond.to_port in element.ports:
                        f_sym = sp.Symbol(f"f{bond.id}")
                        new_eqs.append(sp.Eq(dq_sym, f_sym))
                        if self.debug:
                            print(f"[C] Добавлено уравнение производной: d{q_sym}/dt = {f_sym}")
                        break
        # Для I (p): dp/dt = e_bond
        for element in self.model.elements:
            if element.type == ElementType.INDUCTOR:
                p_sym = sp.Symbol(f"p{element.id}")
                dp_sym = sp.Symbol(f"dotp{element.id}")
                for bond in self.model.bonds:
                    if bond.from_port in element.ports or bond.to_port in element.ports:
                        e_sym = sp.Symbol(f"e{bond.id}")
                        new_eqs.append(sp.Eq(dp_sym, e_sym))
                        if self.debug:
                            print(f"[I] Добавлено уравнение производной: d{p_sym}/dt = {e_sym}")
                        break
        return new_eqs

    def build_cauchy_form(self):
        """
        Генерирует каноническую форму Коши (state equations) для системы.
        Теперь ef_map строится динамически на этапе подстановки!
        """
        # 1) Генерация всех уравнений
        eqgen = EquationGenerator(self.model, debug=self.debug)
        self.global_equations = eqgen.generate_equations()  # только "сырые" алгебраические уравнения

        # 2) Собираем все финальные переменные
        self.final_vars = list()
        self._collect_final_vars()

        # 3) Собираем сырые уравнения Коши для всех q и p
        self.cauchy_equations = []
        self._build_raw_cauchy_eqs()

        # 4) Рекурсивно раскрываем eX/fY только по мере необходимости!
        self._recursive_substitute_cauchy()

        if self.debug:
            print("[CauchyFormGenerator] Final cauchy eq:")
            for eq in self.cauchy_equations:
                print("   ", eq)

        return self.cauchy_equations

    
    def _is_ef_symbol(self, s):
        """
        Return True if symbol looks like e4 or f12
        """
        name = s.name
        if len(name) < 2:
            return False
        c0 = name[0]
        return c0 in ('e','f') and name[1:].isdigit()

    # ----------------------------------------------------------------------
    # ШАГ 2: Сбор финальных переменных
    # ----------------------------------------------------------------------
    def _collect_final_vars(self):
        """
        Собирает все переменные состояния, входы и параметры в self.final_vars.
        """
        for elem in self.model.elements:
            # состояния (обычно qX, pY)
            if hasattr(elem, "state_variable") and elem.state_variable:
                if sp.Symbol(elem.state_variable) not in self.final_vars: self.final_vars.append(sp.Symbol(elem.state_variable))
                if sp.Symbol(elem.state_variable) not in self.state_vars: self.state_vars.append(sp.Symbol(elem.state_variable))
            # входы (обычно SE, SF)
            if hasattr(elem, "input_variable") and elem.input_variable:
                iname = elem.input_variable.lstrip('+')
                if iname not in self.final_vars: self.final_vars.append(sp.Symbol(iname))
                if iname not in self.input_vars: self.input_vars.append(sp.Symbol(elem.input_variable))
            # параметры (обычно R, C, I, TF, GY, ...)
            if hasattr(elem, "parameter") and elem.parameter:
                if iname not in self.final_vars: self.final_vars.append(sp.Symbol(elem.parameter))
    # ----------------------------------------------------------------------
    # ШАГ 3: Формируем сырые уравнения Коши для каждой переменной состояния
    # т.е. только начало уравнений, дальше надо пудет подставлять значения
    # ----------------------------------------------------------------------
    def _build_raw_cauchy_eqs(self):
        """
        Для каждой переменной состояния (q, p) формируем eq типа dotq = ..., dotp = ...
        """
        for elem in self.model.elements:
            # Конденсаторы (q): dotq = f_bond
            if elem.type == ElementType.CAPACITOR and hasattr(elem, "state_variable") and elem.state_variable:
                q = sp.Symbol(elem.state_variable)
                dq = sp.Symbol(f'dot{q.name}')
                # Находим bond, где этот элемент участвует
                for bond in self.model.bonds:
                    if bond.from_port in elem.ports or bond.to_port in elem.ports:
                        f = sp.Symbol(f'f{bond.id}')
                        self.cauchy_equations.append(sp.Eq(dq, f))
                        if self.debug:
                            print(f"[C] {dq} = {f}")
                        break
            # Индукторы (p): dotp = e_bond
            if elem.type == ElementType.INDUCTOR and hasattr(elem, "state_variable") and elem.state_variable:
                p = sp.Symbol(elem.state_variable)
                dp = sp.Symbol(f'dot{p.name}')
                for bond in self.model.bonds:
                    if bond.from_port in elem.ports or bond.to_port in elem.ports:
                        e = sp.Symbol(f'e{bond.id}')
                        self.cauchy_equations.append(sp.Eq(dp, e))
                        if self.debug:
                            print(f"[I] {dp} = {e}")
                        break

    # --- ШАГ 6: Рекурсивная подстановка динамически ---
    @staticmethod
    def expr_symbols_set(expr):
        """
        Возвращает frozenset имён свободных переменных в выражении expr.
        """
        return frozenset(str(s) for s in expr.free_symbols)
    
    def is_self_recursive(self, symbol, expr, chain):
        # Проверяем, есть ли symbol в свободных переменных expr (прямое самовыражение)
        if symbol in expr.free_symbols:
            return True
        # Проверяем косвенное: выражаем через ef-символ, который уже есть в цепочке (замкнутый цикл)
        chain_syms = {sp.Symbol(v) for v, e in chain}
        if any(s in expr.free_symbols for s in chain_syms):
            return True
        return False

    def find_all_expr_for_symbol(self, symbol, used_variants):
        """
        Найти все допустимые выражения для symbol, не используя варианты из used_variants (множество строк-выражений).
        """
        candidates = []
        for eq in self.global_equations:
            if symbol in eq.free_symbols:
                sol = sp.solve(eq, symbol, dict=True)
                if sol and len(sol) > 1:
                    if self.debug: print(f"[WARNING] Для {symbol} в {eq} найдено несколько решений: {sol}")
                    continue
                if sol and len(sol) == 1 and symbol in sol[0]:
                    expr = sol[0][symbol]
                    if (
                        expr is not None
                        and not isinstance(expr, bool)
                        and symbol not in expr.free_symbols
                        and str(expr) not in used_variants
                    ):
                        candidates.append(expr)
        return candidates

    def full_subs(self, expr, chain=None, visited=None, max_iter=20):
        """
        Backtracking раскрытие с защитой от циклов: visited - множество цепочек замен.
        """
        if chain is None:
            chain = []
        if visited is None:
            visited = set()
        # Защита: если мы уже были в такой цепочке — не заходим снова
        state = (str(expr), tuple(chain))
        if state in visited:
            return None
        visited.add(state)

        for _ in range(max_iter):
            ef_syms = [s for s in expr.free_symbols if self._is_ef_symbol(s) and s not in self.final_vars]
            if not ef_syms:
                if self.debug: print("  [SUCCESS] Раскрыто выражение:", expr)
                if self.debug: print("  [CHAIN] -> " + " | ".join(f"{v}:{e}" for v, e in chain))
                return expr
            s = ef_syms[0]
            used = {e for v, e in chain if v == str(s)}
            variants = self.find_all_expr_for_symbol(s, used)
            if not variants:
                if self.debug: print(f"  [FAIL] Нет вариантов раскрытия для {s}, цепочка: {chain}")
                return None
            
            found_valid = False  # Сигнализирует, был ли хоть один допустимый вариант (не цикл)
            for expr_variant in variants:
                if self.is_self_recursive(s, expr_variant, chain):
                    if self.debug: print(f"  [CYCLE] Не подставляем {s} через {expr_variant} (обнаружено зацикливание)")
                    continue
                found_valid = True
                if self.debug: print(f"  [TRY] Пробуем для {s} выражение: {expr_variant}")
                chain.append((str(s), str(expr_variant)))
                expr_next = expr.subs(s, expr_variant)
                result = self.full_subs(expr_next, chain=chain, visited=visited, max_iter=max_iter)
                if result is not None:
                    return result
                chain.pop()
            if not found_valid:
                if self.debug: print(f"  [FAIL] Для {s} нет ни одного допустимого пути раскрытия из-за зацикливания")
                return None
            
        if self.debug: print(f' [ERROR] Достигнут лимит итераций! Лимит {max_iter}')
        return None


    def _recursive_substitute_cauchy(self):
        """
        Для каждого уравнения Коши раскрывает его по backtracking, храня цепочку замен (переменная, выражение).
        """
        new_eqs = []
        for row, eq in enumerate(self.cauchy_equations, 1):
            lhs, rhs = eq.lhs, eq.rhs
            if self.debug: print(f"\n[CAUCHY {row}] Исходное: {lhs} = {rhs}")
            result = self.full_subs(rhs, chain=[], max_iter=20)
            if result is not None:
                if self.debug: print(f"[CAUCHY {row}] После подстановок: {lhs} = {result}\n")
                new_eqs.append(sp.Eq(lhs, result))
            else:
                if self.debug: print(f"[CAUCHY {row}] Не удалось полностью раскрыть!\n")
                new_eqs.append(eq)  # Оставляем исходное, если не удалось раскрыть
                
        self.cauchy_equations = new_eqs
        self.cauchy_equations = [sp.Eq(eq.lhs, sp.expand(eq.rhs)) for eq in self.cauchy_equations]# --- раскрываем скобки во всех уравнениях ---
    
    def get_all_ef_variables(self):
        """
        Возвращает список всех eX/fX переменных, встречающихся во всех исходных уравнениях модели.
        """
        ef_vars = set()
        # Используем все уравнения, сгенерированные EquationGenerator
        eqs = getattr(self, "global_equations", None)
        if eqs is None:
            # Если ещё не сгенерированы — запросить у EquationGenerator
            eqgen = EquationGenerator(self.model, debug=getattr(self, 'debug', False))
            eqs = eqgen.generate_equations()
        for eq in eqs:
            for sym in eq.rhs.free_symbols.union(eq.lhs.free_symbols):
                if self._is_ef_symbol(sym):
                    ef_vars.add(sym)
        return sorted(ef_vars, key=lambda s: str(s))
    
    def generate_output_equations(self, output_vars:str):
        """
        Для каждого выбранного пользователем ef-символа (выхода) строит все полностью раскрытые уравнения (без ef).
        Если найдено несколько вариантов — выводит предупреждение и показывает все.

        Example of usage: cform.generate_output_equations(['e11,f11'])

        """
        output_eqs = []
        user_vars = [var.strip() for var in output_vars.split(",") if var.strip()]
        output_vars = [sp.Symbol(v) for v in user_vars]
        var_list = self.get_all_ef_variables()
        for v in output_vars:
            if v not in var_list:
                print('[ERROR]: U provided incorrect variable (out of variable list in model)')
        for y in output_vars:
            if self.debug: print(f"\n[OUTPUT] Раскрываем выражение для выхода: {y}")

            # Соберём все варианты раскрытия, которые полностью избавлены от ef
            successful_exprs = []

            def collect_all_paths(expr, chain=None, visited=None, max_iter=20):
                # Аналог full_subs, но накапливает все удачные ветки раскрытия
                if chain is None:
                    chain = []
                if visited is None:
                    visited = set()
                # Проверяем: если уже посещали такую пару (expr, chain), не идём снова
                state = (str(expr), tuple(chain))
                if state in visited:
                    return
                visited.add(state)

                ef_syms = [s for s in expr.free_symbols if self._is_ef_symbol(s) and s not in self.final_vars]
                if not ef_syms:
                    successful_exprs.append(expr)
                    return
                s = ef_syms[0]
                used = {e for v, e in chain if v == str(s)}
                variants = self.find_all_expr_for_symbol(s, used)
                for expr_variant in variants:
                    if self.is_self_recursive(s, expr_variant, chain):
                        continue
                    chain.append((str(s), str(expr_variant)))
                    expr_next = expr.subs(s, expr_variant)
                    collect_all_paths(expr_next, chain=chain, visited=visited, max_iter=max_iter)
                    chain.pop()
                return

            # Запускаем обход по всем путям раскрытия
            collect_all_paths(y, chain=[], visited=set(), max_iter=20)

            if not successful_exprs:
                if self.debug: print(f"[WARNING] Не удалось полностью раскрыть {y} (нет ни одного пути без ef)")
                continue

            if len(successful_exprs) > 1:
                if self.debug: print(f"[MULTI] Для {y} найдено {len(successful_exprs)} вариантов полного раскрытия:")
                for idx, e in enumerate(successful_exprs):
                    if self.debug: print(f"  Вариант {idx+1}: {e}")

            # Возьмём первый вариант как основной (остальные можно вернуть пользователю отдельно)
            output_eqs.append(sp.Eq(y, successful_exprs[0]))

        self.output_vars = output_vars
        self.output_eqs = [sp.Eq(eq.lhs, sp.expand(eq.rhs)) for eq in output_eqs]# --- раскрываем скобки во всех уравнениях ---
        
        return self.output_eqs

    def interactive_generate_output_equations(self):
        """
        Интерактивный выбор выходных переменных из всех ef, ввод через запятую.
        Оболочка для пользователя.
        """
        ef_candidates = self.get_all_ef_variables()
        if self.debug: print("Доступные eX/fX переменные для выбора в качестве выхода:")
        if self.debug: print(", ".join(str(v) for v in ef_candidates))
        s = input("Введите через запятую (например, e4,f7): ")
        user_vars = [var.strip() for var in s.split(",") if var.strip()]
        output_vars = [sp.Symbol(v) for v in user_vars]
        if self.debug: print(f"Вы выбрали выходные переменные: {output_vars}")
        # Генерируем уравнения выходов
        output_eqs = self.generate_output_equations(output_vars)
        # Можно сразу вывести результат:
        if self.debug: print("\nИтоговые уравнения выходов:")
        for eq in output_eqs:
            print(eq)
        # Вернуть список для дальнейшей работы
        return output_eqs



class StateSpaceBuilder:
    """
    Builds a matrix form x_dot = A x + B u from the cauchy-form equations
    provided by CauchyFormGenerator. It does not inherit from that class,
    but uses it as a helper.

    IT works only for ODE!!!

    """
    #TODO DAE support for non invertable M matrix

    def __init__(self, model, cform_gen:CauchyFormGenerator, debug:bool = False):
        self.model = model
        self.cform_gen = cform_gen
        self.debug = debug

        # Final results
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        # self.y
        
        self.x_vars = []     # list of sympy.Symbol for states
        self.u_vars = []     # list of sympy.Symbol for inputs
        
        # Will store the cauchy equations (each is sympy.Eq)
        self.state_equations = []

    def build_state_space(self):
        """
        1) Generate cauchy-form eqs via CauchyFormGenerator (already substituted),
        2) Identify states x and inputs u,
        3) Convert to linear matrix form using sympy.
        """
        # Используем данные из cform_gen:
        self.state_equations: List[sp.Eq] = self.cform_gen.cauchy_equations
        self.x_vars: List[sp.Symbol] = self.cform_gen.state_vars
        self.u_vars: List[sp.Symbol] = self.cform_gen.input_vars
        self.output_vars: List[sp.Symbol] = self.cform_gen.output_vars
        self.output_eqs: List[sp.Eq] = self.cform_gen.output_eqs

        if self.debug:
            print("[StateSpaceBuilder] state_equations (Cauchy) =")
            for eq in self.state_equations:
                print("  ", eq)
            print("[StateSpaceBuilder] x_vars =", self.x_vars)
            print("[StateSpaceBuilder] u_vars =", self.u_vars)

        # 3) Convert to matrix form
        self._convert_to_matrix_form()

        # 4. Формируем матрицы C и D (выходы — по умолчанию состояния)
        if self.debug:
            print("x_vars:", self.x_vars)
            print("u_vars:", self.u_vars)
            print("output_vars:", self.output_vars)
            for eq in self.output_eqs:
                print("eq.lhs:", eq.lhs, " | eq.rhs:", eq.rhs)

        if self.output_vars is None or self.output_eqs is None:
            if self.debug: 
                print("[ERROR] Не заданы выходные переменные или уравнения! Используйте generate_output_equations.")
            return
        
        m_size = len(self.output_vars)
        n_state = len(self.x_vars)
        n_input = len(self.u_vars)
        
        if self.debug:
            print(f'System has {m_size} outputs, {n_state} states and {n_input} inputs(including constant inputs)')

        C_matrix = sp.zeros(m_size, n_state)
        D_matrix = sp.zeros(m_size, n_input)

        print('C and D matrix values :: ', C_matrix, D_matrix)

        for i, y in enumerate(self.output_vars):
            # Находим уравнение для y
            eq_found = False
            for eq in self.output_eqs:
                if eq.lhs == y:
                    rhs = eq.rhs
                    collected = sp.collect(sp.expand(rhs), list(self.x_vars) + list(self.u_vars))

                    for j, x in enumerate(self.x_vars):
                        c = collected.coeff(x)
                        if self.debug: print(f"  coeff({x}) в выражении {collected}: {c}")
                        C_matrix[i, j] = c
                    for j, u in enumerate(self.u_vars):
                        d = collected.coeff(u)
                        if self.debug: print(f"  coeff({u}) в выражении {collected}: {d}")
                        D_matrix[i, j] = d
                    eq_found = True
                    break
            if not eq_found:
                if self.debug: print(f"[ERROR] Не найдено уравнение для выхода {y}, строка будет нулевая.")
        self.C = C_matrix
        self.D = D_matrix
        if self.debug:
            print("C =", C_matrix)
            print("D =", D_matrix)

        return (self.A, self.B, self.C, self.D)
    # -------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------

    def _convert_to_matrix_form(self):
        """
        The cauchy eqs are something like:
          d(q2)/dt = expression_in(x,u)
          d(p4)/dt = expression_in(x,u)
        We'll parse them into a system eq=0 and use linear_eq_to_matrix
        or a manual approach to get A, B.
        """

        # Build eq_zero_forms: d(x_i)/dt - RHS = 0
        eq_list = []
        dx_syms = []  # each d(x)/dt as a Symbol

        for eq in self.state_equations:
            lhs, rhs = eq.lhs, eq.rhs
            # lhs typically is d(q2)_dt, a Function or Symbol
            # we transform that into a Symbol, e.g. dq2dt
            # let's define a mapping
            if lhs.is_Function: 
                # e.g. d(q2)_dt(...), we can do
                sname = lhs.func.__name__  # e.g. 'd(q2)_dt'
                dx_var = sp.Symbol(sname)
            elif lhs.is_Symbol:
                dx_var = lhs
            else:
                # fallback
                dx_var = sp.Symbol(f"dx_{lhs}")

            eq_list.append(dx_var - rhs)  # => dx_var - expr = 0
            dx_syms.append(dx_var)

        # Now we have a system eq_list = [  dx_var_i - expr_i, ... ] = 0
        # We'll define a set of all variables: dx_syms + x_vars + u_vars
        all_vars = list(dx_syms) + list(self.x_vars) + list(self.u_vars)

        # Use linear_eq_to_matrix if we assume linear system
        lhs_mat, rhs_mat = sp.linear_eq_to_matrix(eq_list, all_vars)
        # The system => lhs_mat * all_vars = rhs_mat
        # Typically in a purely linear BG, we expect rhs=0 or constant terms

        n = len(dx_syms)
        # M = lhs_mat[:, :n] # block for dx
        # Ablock = lhs_mat[:, n:n+len(self.x_vars)]
        # Ublock = lhs_mat[:, n+len(self.x_vars): n+len(self.x_vars)+len(self.u_vars)]

        # But in many bond-graph approaches, we do:
        # M * dx + Ablock*x + Ublock*u = 0 => dx = - M^-1(A x + B u)
        # We'll do something similar:

        # check shape
        rows, cols = lhs_mat.shape
        if rows != n:
            # The number of equations doesn't match number of states => might be partial or nonlinear
            if self.debug:
                print("[WARNING] Not a perfect linear system. eq rows != # states.")
            self.A = None
            self.B = None
            return

        # Extract blocks
        dx_block = lhs_mat[:, :n]  # block for the dx syms
        x_block = lhs_mat[:, n : n + len(self.x_vars)]
        u_block = lhs_mat[:, n + len(self.x_vars) : n + len(self.x_vars) + len(self.u_vars)]
        # remainder => constants, if exist => rhs_mat ?

        # We'll define: M = dx_block, so M * dx + x_block*x + u_block*u = rhs_mat
        # => M * dx = rhs_mat - (x_block*x + u_block*u)
        # => dx = M^-1 * (rhs_mat - x_block*x - u_block*u)

        # Usually in a pure BG system, rhs_mat is all zeros, but let's not assume.

        # In typical linear BG: M is identity or near. We do a general approach:
        if dx_block.det() == 0:
            if self.debug:
                print("[WARNING] dx_block is singular => can't invert => possibly partial or nonlinear system.")
            self.A = None
            self.B = None
            return

        M_inv = dx_block.inv()
        # => dx = M_inv * (rhs_mat - x_block*x - u_block*u)
        # => dx = -M_inv * x_block * x - M_inv * u_block * u + M_inv*rhs_mat
        # So => dx = A x + B u + const
        # Let's define A = - M_inv * x_block, B = - M_inv * u_block
        # ignoring constant for now

        A_mat = -M_inv * x_block
        B_mat = -M_inv * u_block

        self.A = A_mat
        self.B = B_mat

        if self.debug:
            print("[StateSpaceBuilder] A =", self.A)
            print("[StateSpaceBuilder] B =", self.B)
            print("[StateSpaceBuilder] RHS =", rhs_mat)
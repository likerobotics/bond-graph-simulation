# equations/cauchy_form.py

import sympy as sp
from bond_graph_simulation.equations.generator import EquationGenerator
from bond_graph_simulation.core.base import ElementType

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
        self.state_cache = {} # for expressitons

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
                dq_sym = sp.Symbol(f"dotq{element.id}")  # derivative symbol
                # Find the bond connected to this element
                for bond in self.model.bonds:
                    if bond.from_port in element.ports or bond.to_port in element.ports:
                        f_sym = sp.Symbol(f"f{bond.id}")
                        new_eqs.append(sp.Eq(dq_sym, f_sym))
                        if self.debug:
                            print(f"[C] Added equation for state derivative : d{q_sym}/dt = {f_sym}")
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
                            print(f"[I] Added equation for state derivative : d{p_sym}/dt = {e_sym}")
                        break
        return new_eqs

    def build_cauchy_form(self):
        """
        Generates the Cauchy canonical form (state equations) for the system.
        ef_map is built dynamically during substitution!
        """
        # 1) Generate all bond graph equations
        eqgen = EquationGenerator(self.model, debug=self.debug)
        self.global_equations = eqgen.generate_equations()  # только "сырые" алгебраические уравнения

        # 2) Collect final variables (states, inputs, parameters)
        self.final_vars = list()
        self._collect_final_vars()
        print("self.final_vars = ", self.final_vars)

        # 3) Create raw Cauchy equations for state derivatives
        self.cauchy_equations = []
        self._build_raw_cauchy_eqs()
        
        print("START RECURECIEVE SUBS")
        # 4) Recurcivly substitute eX/fY only when needed!
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

    def _collect_final_vars(self):
        """
        Collects all state variables, inputs, and parameters into self.final_vars.
        """
        for elem in self.model.elements:
            # state (usally qX, pY)
            if hasattr(elem, "state_variable") and elem.state_variable:
                if sp.Symbol(elem.state_variable) not in self.final_vars: self.final_vars.append(sp.Symbol(elem.state_variable))
                if sp.Symbol(elem.state_variable) not in self.state_vars: self.state_vars.append(sp.Symbol(elem.state_variable))
            # inputs (usually SE, SF)
            if hasattr(elem, "input_variable") and elem.input_variable:
                iname = elem.input_variable.lstrip('+')
                if iname not in self.final_vars: self.final_vars.append(sp.Symbol(iname))
                if iname not in self.input_vars: self.input_vars.append(sp.Symbol(elem.input_variable))
            # parameters (usualyy R, C, I, TF, GY, ...)
            if hasattr(elem, "parameter") and elem.parameter:
                if iname not in self.final_vars: self.final_vars.append(sp.Symbol(elem.parameter))
    
    def _build_raw_cauchy_eqs(self):
        """
        For each state variable (q, p) form equations like
        dotq = ... 
        dotp = ...
        i.e. inisial equation for beginings, further substitution is needed.
        """
        for elem in self.model.elements:
            # Capacitive (q): dotq = f_bond
            if elem.type == ElementType.CAPACITOR and hasattr(elem, "state_variable") and elem.state_variable:
                q = sp.Symbol(elem.state_variable)
                dq = sp.Symbol(f'dot{q.name}')
                # Find the bond connected to this element
                for bond in self.model.bonds:
                    if bond.from_port in elem.ports or bond.to_port in elem.ports:
                        f = sp.Symbol(f'f{bond.id}')
                        self.cauchy_equations.append(sp.Eq(dq, f))
                        if self.debug:
                            print(f"[C] {dq} = {f}")
                        break
            # Inductive (p): dotp = e_bond
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

    # --- ШАГ 6: Recursive substitution ---
    @staticmethod
    def expr_symbols_set(expr):
        """
        Returns frozenset of free variable names in expr.
        """
        return frozenset(str(s) for s in expr.free_symbols)
    
    def is_self_recursive(self, symbol, expr, chain):
        # Check for self-recursion in substitution chain(is symbol defined through itself)
        if symbol in expr.free_symbols:
            return True
        # Check for indirect recursion: express through ef-symbol already in chain (closed loop)
        chain_syms = {sp.Symbol(v) for v, e in chain}
        if any(s in expr.free_symbols for s in chain_syms):
            return True
        return False

    def find_all_expr_for_symbol(self, symbol, used_variants):
        """
        Find all valid expressions for symbol, not using variants from used_variants (set of string expressions).
        """
        candidates = []
        for eq in self.global_equations:
            if symbol in eq.free_symbols:
                sol = sp.solve(eq, symbol, dict=True)
                if sol and len(sol) > 1:
                    print(f"[WARNING] [Critical] For {symbol} in {eq} found multiple solutions: {sol}")
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
    
    def is_ef(self, sym: sp.Symbol) -> bool:
        """
        Check whether a symbol is of type e* or f*.
        These variables must be recursively unfolded.
        """
        return sym.name[0] in ('e', 'f')

    def solve_single(self, var, visited):
        """
        Extract all possible expressions of the form:

            var = expression

        from the given equation list.
        This includes:
        • direct equations (Eq(var, expr))
        • reversed equations (Eq(expr, var))
        • implicitly solvable equations (using sympy.solve)

        Returns list of candidate expressions for var.
        """
        candidates = []

        for eq in self.global_equations:
            lhs, rhs = eq.lhs, eq.rhs

            # Direct form: var = RHS
            if lhs == var:
                candidates.append(rhs)
                continue

            # Reversed form: LHS = var  → var = LHS
            if rhs == var:
                candidates.append(lhs)
                continue

            # If var appears anywhere, try solving symbolically
            if var in eq.free_symbols:
                try:
                    sols = sp.solve(eq, var, dict=False)
                    if sols:
                        candidates.append(sols[0])
                except Exception:
                    pass  # not solvable; ignore

        return candidates

    def expand_expr(self, expr, visited):
        """
        Recursively expand all e* and f* symbols inside an expression.
        Parameters inside the expression remain untouched.

        Example:
            If expr = e4 + e9 - SE1
            then it recursively unfolds e4 and e9.
        """
        new_expr = expr

        for s in list(expr.free_symbols):
            if self.is_ef(s):
                # Recursively compute substitution for s
                sub = self.substitute_until_final(s, visited)
                new_expr = sp.simplify(new_expr.subs(s, sub))

        return sp.simplify(new_expr)


    def substitute_until_final(self, target_var, visited=None):
        """
        self.global_equations - all equations list
        target - sympy.Symbol to express through final_vars
        Returns expression for target in terms of final_vars, or None if not possible.  
        Uses iterative substitution with tracking of defined variables.
        """

                
        if not self.is_ef(target_var):
            return target_var # If var is not an e* or f* → it is a parameter → return as-is

        # Initialize cycle-detection set
        if visited is None:
            visited = set()

        # Cycle detected
        if target_var in visited:
            raise RuntimeError(f"Cycle while expanding {target_var}")

        # Add this variable to visited chain
        visited = visited | {target_var}

        # Get all possible var = expr candidates
        candidates = self.solve_single(target_var, visited)
        if not candidates:
            raise RuntimeError(f"Variable {target_var} cannot be solved from given equations.")

        # Rank candidates:
        # Fewer (e*, f*) inside expression → higher priority
        scored = []
        for expr in candidates:
            ef_vars = [s for s in expr.free_symbols if self.is_ef(s)]
            score = len(ef_vars)
            scored.append((score, expr))

        scored.sort(key=lambda x: x[0])

        # Try each candidate until one succeeds
        last_err = None
        for score, expr in scored:
            try:
                result = self.expand_expr(expr, visited)
                return result  # success
            except RuntimeError as e:
                last_err = e
                continue

        # If all candidates failed, throw last error
        if last_err:
            raise last_err

        raise RuntimeError(f"Cannot expand {target_var}")
        #self.final_vars 
        
    def full_subs(self, expr, chain=None, visited=None, fail_cache=None, depth=0, max_iter=20):
        """
        Backtracking раскрытие EF-переменных с подробным логированием.
        depth — глубина рекурсии (для форматирования вывода).
        """
        pad = "  " * depth  # отступ

        if fail_cache is None:
            fail_cache = set()
        if chain is None:
            chain = []
        if visited is None:
            visited = set()

        print(f"{pad}▶ full_subs(depth={depth}) expr = {expr} chain = {chain}")

        # visited — мягкая версия (только expr)
        state = str(expr)
        if state in visited:
            # print(f"{pad}   visited hit → {state}")
            return None
        visited.add(state)
        print(f"{pad}  visited = {visited}")

        for iter_num in range(max_iter):
            print(f"{pad}  --- iter {iter_num} ---")

            # 1. Поиск ef-символов
            ef_syms = [
                s for s in expr.free_symbols
                if self._is_ef_symbol(s) and s not in self.final_vars
            ]

            # 2. Если раскрывать нечего
            if not ef_syms:
                print(f"{pad}  ✔ SUCCESS: раскрыто выражение {expr}")
                print(f"{pad}  ✔ CHAIN = {chain}")
                return expr

            # 3. Берём первый ef-символ
            s = ef_syms[0]
            print(f"{pad}  ef_syms = {ef_syms} chosen symbol = {s}")

            # уже использованные подстановки для данного символа
            used = {e for v, e in chain if v == str(s)}
            

            # 4. Поиск вариантов раскрытия
            variants = self.find_all_expr_for_symbol(s, used)

            print(f"{pad}  variants = {variants} used_variants = {used}")
            if not variants:
                print(f"{pad}  !!!! Нет вариантов раскрытия для {s}. STOP.")
                return None

            found_valid = False

            # 5. Перебираем все варианты раскрытия
            for expr_variant in variants:
                fail_key = (str(s), str(expr_variant), str(expr))
                print(f"{pad}   → try variant: {s} = {expr_variant}")
                print(f"{pad}     fail_key = {fail_key}")

                if fail_key in fail_cache:
                    print(f"{pad}     !!! skip: in fail_cache")
                    continue

                if self.is_self_recursive(s, expr_variant, chain):
                    print(f"{pad}     !!! skip: self-recursive")
                    continue

                found_valid = True
                chain.append((str(s), str(expr_variant)))
                print(f"{pad}     ✓ chain PUSH → {chain}")

                expr_next = expr.subs(s, expr_variant)
                print(f"{pad}     substituted expr_next = {expr_next}")

                # рекурсивный вызов
                result = self.full_subs(
                    expr_next,
                    chain=chain,
                    visited=visited,
                    fail_cache=fail_cache,
                    depth=depth + 1,
                    max_iter=max_iter
                )

                if result is not None:
                    print(f"{pad}  ✔ RETURN from depth {depth}: {result}")
                    return result

                # backtrack
                chain.pop()
                print(f"{pad}     ↩ chain POP → {chain}")

            # 6. Если ни один вариант не допустим
            if not found_valid:
                print(f"{pad}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! all variants invalid, updating fail_cache")
                for expr_variant in variants:
                    fail_cache.add((str(s), str(expr_variant), str(expr)))
                print(f"{pad}  fail_cache = {fail_cache}")
                return None

        print(f"{pad}!!! ERROR: max_iter={max_iter} reached")
        return None



    def _recursive_substitute_cauchy(self):
        """
        Для каждого уравнения Коши раскрывает его по backtracking, храня цепочку замен (переменная, выражение).
        """
        new_eqs = []
        for row, eq in enumerate(self.cauchy_equations, 1):
            lhs, rhs = eq.lhs, eq.rhs
            print(f"\n[CAUCHY STATE {row}] Исходное: {lhs} = {rhs}")

            ### RECURSIVE SUBSTITUTION!!!
            result = self.substitute_until_final(rhs)

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
        For each user-selected ef-symbol (output), builds fully expanded equations (without ef).
        If multiple variants found — warns and shows all.
        
        Example of usage: cform.generate_output_equations(['e11,f11'])
        """
        print("Debug state:", self.debug)

        output_eqs = []
        user_vars = [var.strip() for var in output_vars.split(",") if var.strip()]
        print("Entered list", user_vars)
        output_vars = [sp.Symbol(v.strip()) for v in user_vars]
        var_list = self.get_all_ef_variables()
        for v in output_vars:
            if v not in var_list:
                print('[ERROR]: U provided incorrect variable (out of variable list in model)')
        print("symb entered list", output_vars)
        output_eqs = []

        for var in output_vars:
            print(f"\n[OUTPUT] ROW Eq for output variable: {var} [{type(var)}]")
            found = False
            for eq in self.global_equations:
                # 1. Direct Eq(var, ...)
                if eq.lhs == var:
                    output_eqs.append(sp.Eq(var, eq.rhs))
                    found = True
                    break
                # 2. Direct Eq(..., var)
                if eq.rhs == var:
                    output_eqs.append(sp.Eq(var, eq.lhs))
                    found = True
                    break
                # 3. General: try solve for var if it is present in eq
                if var in eq.free_symbols:
                    try:
                        sols = sp.solve(eq, var, dict=False)
                        if sols:
                            output_eqs.append(sp.Eq(var, sols[0]))
                            found = True
                            break
                    except Exception:
                        pass
            if not found:
                print(f"[OUTPUT EQ ERROR] FAILED FOR: {var}")
        print("\n[INFO] Output equations:")
        for eq in output_eqs:
            print(eq)

        self.output_eqs = output_eqs
        # ROW OUTPUT EQUASTION PREPARATION DONE
        # SIMILAR TO _recursive_substitute_cauchy
        """
        For each output equation open here its backtracking, storing the chain of substitutions (variable, expression).
        """
        new_eqs = []
        for row, eq in enumerate(self.output_eqs, 1):
            # if self.debug: 
            print(f"\n[OUTPUT] Generating output equation for: {row}")
            lhs, rhs = eq.lhs, eq.rhs
            # if self.debug: 
            print(f"\n[CAUCHY OUT {row}] Initial: {lhs} = {rhs}")

            ### RECURSIVE SUBSTITUTION!!! FOR EACH COMPONENT IN RHS
            my_subs = {}
            for s in rhs.free_symbols:
                if self.is_ef(s):
                    result = self.substitute_until_final(s)
                    my_subs[s]= result

            print("my_subs:",my_subs)
            
            # result = self.substitute_until_final(rhs)
            rhs_substituted = rhs.subs(my_subs)

            if rhs_substituted is not None:
                if self.debug: print(f"[CAUCHY OUT {row}] After substitutions: {lhs} = {rhs_substituted}\n")
                new_eqs.append(sp.Eq(lhs, rhs_substituted))
            else:
                if self.debug: print(f"[CAUCHY OUT {row}] Could not fully expand!\n")
                new_eqs.append(eq)  # Leave original if expansion failed
                
            self.output_eqs = new_eqs
            self.output_eqs = [sp.Eq(eq.lhs, sp.expand(eq.rhs)) for eq in self.output_eqs]# --- OPEN BRAKETS IN ALL EQ ---


        self.output_vars = output_vars

        if len(output_vars) != len(output_eqs):
            print(f"RESULT!!! Requested {output_vars} MISMATCH !!!! found {output_eqs} ")

        return self.output_eqs

    def interactive_generate_output_equations(self):
        """
        Interactive selection of output variables from all ef, comma-separated input.
        User-friendly wrapper.
        """
        ef_candidates = self.get_all_ef_variables()
        if self.debug: print("Доступные eX/fX переменные для выбора в качестве выхода:")
        if self.debug: print(",".join(str(v) for v in ef_candidates))
        output_vars = input("Введите через запятую (например, e4,f7): ")
        
        # Generate output equations
        output_eqs = self.generate_output_equations(output_vars)
        # Print final output equations
        if self.debug: print("\nResulting output equations:")
        for eq in output_eqs:
            print(eq)
        # Return list for further processing
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
                print("[ERROR] Outputvariablesor equations are not provided! USE generate_output_equations.")
            return
        
        m_size = len(self.output_vars)
        n_state = len(self.x_vars)
        n_input = len(self.u_vars)
        
        if self.debug:
            print(f'System has {m_size} outputs, {n_state} states and {n_input} inputs(including constant inputs)')

        C_matrix = sp.zeros(m_size, n_state)
        D_matrix = sp.zeros(m_size, n_input)

        for i, y in enumerate(self.output_vars):
            # Находим уравнение для y
            eq_found = False
            for eq in self.output_eqs:
                if eq.lhs == y:
                    rhs = eq.rhs
                    collected = sp.collect(sp.expand(rhs), list(self.x_vars) + list(self.u_vars))

                    for j, x in enumerate(self.x_vars):
                        c = collected.coeff(x)
                        if self.debug: print(f"  coeff({x}) detected in {collected}: {c}")
                        C_matrix[i, j] = c
                    for j, u in enumerate(self.u_vars):
                        d = collected.coeff(u)
                        if self.debug: print(f"  coeff({u}) detected in {collected}: {d}")
                        D_matrix[i, j] = d
                    eq_found = True
                    break
            if not eq_found:
                if self.debug: print(f"[ERROR] No equation found for output {y}, row will be zero.")
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
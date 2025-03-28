import numpy as np
import random
import matplotlib.pyplot as plt


class CanalBSC:
    """
    Simula um Canal Binário Simétrico (BSC) com probabilidade p de erro.
    """
    def __init__(self, p):
        """
        Inicializa o canal com probabilidade p de erro.
        
        Args:
            p: Probabilidade de inverter um bit
        """
        self.p = p
    
    def transmitir(self, palavra):
        """
        Transmite uma palavra através do canal, com possibilidade de erro.
        
        Args:
            palavra: Array ou lista de bits a ser transmitida
            
        Returns:
            Array com a palavra possivelmente modificada
        """
        resultado = np.copy(palavra)
        for i in range(len(resultado)):
            if random.random() < self.p:
                resultado[i] = 1 - resultado[i]  # Inverte o bit
        return resultado

class CodificadorHamming:
    """
    Implementa um codificador de Hamming usando matriz geradora G.
    """
    def __init__(self, G):
        """
        Inicializa o codificador com a matriz geradora G.
        
        Args:
            G: Matriz geradora para o código Hamming
        """

        self.G = np.array(G)
        self.k = self.G.shape[0]  # Dimensão da palavra de entrada
        self.n = self.G.shape[1]  # Dimensão da palavra de saída
    
    def codificar(self, u):
        """
        Codifica a palavra de informação u em palavra de código v.
        
        Args:
            u: Palavra de informação (vetor de k bits)
            
        Returns:
            Palavra codificada v (vetor de n bits)
        """
        # Multiplicação de u por G (módulo 2)
        v = np.dot(u, self.G) % 2
        return v

class DecodificadorHamming:
    """
    Implementa um decodificador de Hamming usando matriz de verificação H^T.
    """
    def __init__(self, H_T, G):
        """
        Inicializa o decodificador com a matriz de verificação transposta H^T.
        
        Args:
            H_T: Matriz de verificação transposta H^T
            G: Matriz geradora (necessária para recuperar u a partir de v)
        """
        self.H_T = np.array(H_T)
        self.G = np.array(G)
        self.n = self.H_T.shape[0]  # Comprimento da palavra codificada  ###### TO DO: Verificar se é a dimensão da matriz H_T
        
        # Cria o mapeamento de síndrome para padrão de erro
        self.mapa_sindromes = self._criar_mapa_sindromes()
    
    def _criar_mapa_sindromes(self):
        """
        Cria um mapeamento completo de síndromes para padrões de erro de menor peso.
        Percorre todos os padrões de erro possíveis, do menor para o maior peso,
        e associa cada síndrome ao primeiro (menor peso) padrão de erro que a gera.
        
        Returns:
            Dicionário mapeando síndrome (como tupla) para padrão de erro
        """
        mapa = {}
        
        # Número de bits na palavra codificada
        n = self.n
        # Número de bits na síndrome (n-k)
        n_k = self.H_T.shape[1]
        
        # Itera sobre todos os pesos possíveis (0, 1, 2, ..., n)
        for peso in range(n + 1):
            # Gera todos os padrões de erro com o peso atual
            for indices_erro in self._combinacoes(range(n), peso):
                # Cria o padrão de erro
                e = np.zeros(n, dtype=int)
                for i in indices_erro:
                    e[i] = 1
                    
                # Calcula a síndrome para este padrão de erro
                sindrome = tuple(np.dot(e, self.H_T) % 2)
                
                # Se esta síndrome ainda não foi mapeada, associa ao padrão de erro atual
                if sindrome not in mapa:
                    mapa[sindrome] = e
                    
            # Se já mapeamos todas as 2^(n-k) síndromes possíveis, podemos parar
            if len(mapa) == 2**n_k:
                break
        
        return mapa

    def _combinacoes(self, elementos, k):
        """
        Gera todas as combinações de k elementos a partir de uma lista de elementos.
        Usado para gerar todos os padrões de erro com um determinado peso.
        
        Args:
            elementos: Lista de elementos (índices dos bits)
            k: Número de elementos em cada combinação (peso do erro)
        
        Returns:
            Lista de combinações, onde cada combinação é uma tupla de índices
        """
        if k == 0:
            return [()]
        
        if not elementos:
            return []
        
        primeiro, resto = elementos[0], elementos[1:]
        
        # Combinações que incluem o primeiro elemento
        com_primeiro = [(primeiro,) + comb for comb in self._combinacoes(resto, k-1)]
        
        # Combinações que não incluem o primeiro elemento
        sem_primeiro = self._combinacoes(resto, k)
        
        return com_primeiro + sem_primeiro
    
    def decodificar(self, r):
        """
        Decodifica a palavra recebida r.
        
        Args:
            r: Palavra recebida (potencialmente com erros)
            
        Returns:
            Tupla (u, v, e) onde:
            - u é a palavra de informação decodificada
            - v é a palavra codificada recuperada
            - e é o padrão de erro identificado
        """
        # Calcula a síndrome
        sindrome = tuple(np.dot(r, self.H_T) % 2)
        
        # Encontra o padrão de erro associado à síndrome
        if sindrome in self.mapa_sindromes:
            e = self.mapa_sindromes[sindrome]
        else:
            # Se a síndrome não estiver no mapa, assume erro não corrigível
            print("Erro não corrigível: síndrome desconhecida")
            e = np.zeros(self.n, dtype=int)
        
        # Recupera a palavra codificada v = r + e (XOR)
        v = (r + e) % 2
        
        # Recupera a palavra de informação u (primeiros k bits de v, para códigos sistemáticos)
        k = self.G.shape[0]
        u = v[:k]
        
        return u, v, e

def exemplo_hamming_74():
    """
    Exemplo de uso com código Hamming (7,4)
    """
    # Matriz geradora G para código Hamming (7,4) na forma sistemática
    G = np.array([
        [1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1]
    ])
    
    # Matriz de verificação transposta H^T
    H_T = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Palavra de informação
    u = np.array([1, 0, 1, 1])
    
    # Codificação
    codificador = CodificadorHamming(G)
    v = codificador.codificar(u)
    print(f"Palavra de informação (u): {u}")
    print(f"Palavra codificada (v): {v}")
    
    # Transmissão pelo canal com erro
    p_erro = 0.1
    canal = CanalBSC(p_erro)
    r = canal.transmitir(v)
    print(f"Palavra recebida (r): {r}")
    
    # Decodificação
    decodificador = DecodificadorHamming(H_T, G)
    u_recuperado, v_recuperado, erro = decodificador.decodificar(r)
    print(f"Padrão de erro detectado (e): {erro}")
    print(f"Palavra codificada recuperada (v): {v_recuperado}")
    print(f"Palavra de informação recuperada (u): {u_recuperado}")
    
    return u, v, r, u_recuperado, v_recuperado, erro




def exemplo_hamming_95():
    """
    Exemplo de uso com código Hamming (9,5)
    """
    # Matriz geradora G para código Hamming (9,5) na forma sistemática
    G = np.array([
        [1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 1, 1]
    ])
    
    # Matriz de verificação transposta H^T
    H_T = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Palavra de informação
    u = np.array([1, 0, 1, 1, 0])
    
    # Codificação
    codificador = CodificadorHamming(G)
    v = codificador.codificar(u)
    print(f"Palavra de informação (u): {u}")
    print(f"Palavra codificada (v): {v}")
    
    # Transmissão pelo canal com erro
    p_erro = 0.1
    canal = CanalBSC(p_erro)
    r = canal.transmitir(v)
    print(f"Palavra recebida (r): {r}")
    
    # Decodificação
    decodificador = DecodificadorHamming(H_T, G)
    u_recuperado, v_recuperado, erro = decodificador.decodificar(r)
    print(f"Padrão de erro detectado (e): {erro}")
    print(f"Palavra codificada recuperada (v): {v_recuperado}")
    print(f"Palavra de informação recuperada (u): {u_recuperado}")
    
    return u, v, r, u_recuperado, v_recuperado, erro

def simular_transmissao(G, H_T, erro_canal, num_bits):
    """
    Função auxiliar que realiza a simulação de transmissão para um código específico.
    """
    k = G.shape[0]  # Tamanho do grupo de bits (informação)
    
    # Garantir que o número de bits seja múltiplo de k
    num_palavras = num_bits // k
    num_bits = num_palavras * k
    
    print(f"Gerando {num_bits} bits de informação aleatórios...")
    
    # Gerar bits de informação aleatórios
    bits_originais = np.random.randint(0, 2, num_bits)
    palavras_originais = bits_originais.reshape(num_palavras, k)
    
    # Inicializar codificador e decodificador
    codificador = CodificadorHamming(G)
    decodificador = DecodificadorHamming(H_T, G)
    
    resultados = []
    
    for p in erro_canal:
        print(f"\nSimulando transmissão com probabilidade de erro p = {p}")
        canal = CanalBSC(p)
        bits_errados = 0
        
        for i in range(num_palavras):
            u = palavras_originais[i]
            v = codificador.codificar(u)
            r = canal.transmitir(v)
            u_recuperado, _, _ = decodificador.decodificar(r)
            
            for j in range(k):
                if u[j] != u_recuperado[j]:
                    bits_errados += 1
            
            if (i + 1) % (num_palavras // 10) == 0:
                percentual = (i + 1) * 100 // num_palavras
                # print(f"Progresso: {percentual}% ({i + 1}/{num_palavras} palavras processadas)")
        
        prob_erro = bits_errados / num_bits
        resultados.append((p, prob_erro))
        print(f"Probabilidade de erro para p = {p}: {prob_erro:.8f} ({bits_errados} bits errados de {num_bits})")
    
    return resultados

def simular_sem_codigo(erro_canal, num_bits):
    """
    Simula transmissão sem codificação (y=x).
    """
    return [(p, p) for p in erro_canal]

def plotar_comparacao(resultados_95, resultados_74, resultados_sem_codigo):
    """
    Plota os três gráficos para comparação.
    """
    plt.figure(figsize=(12, 8))
    
    # Extrair valores para cada conjunto de resultados
    p_95, prob_95 = zip(*resultados_95)
    p_74, prob_74 = zip(*resultados_74)
    p_sem, prob_sem = zip(*resultados_sem_codigo)
    
    # Plotar os três gráficos
    plt.loglog(p_95, prob_95, 'o-', label='Código Hamming (9,5)', color='blue')
    plt.loglog(p_74, prob_74, 's-', label='Código Hamming (7,4)', color='red')
    plt.loglog(p_sem, prob_sem, '--', label='Sem codificação', color='green')
    
    plt.gca().invert_xaxis()  # Inverte o eixo x
    plt.grid(True, which="both", ls="-")
    plt.xlabel('Probabilidade de erro do canal (p)')
    plt.ylabel('Probabilidade de erro de bit')
    plt.title('Comparação das taxas de erro de bit')
    plt.legend()
    plt.savefig('comparacao_codigos.png')
    plt.show()

def funcao_final():
    """
    Realiza simulação comparativa entre diferentes códigos de Hamming e sem codificação.
    """
    # Matriz G e H_T para código Hamming (9,5)
    G_95 = np.array([
        [1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 1, 1]
    ])
    
    H_T_95 = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Matriz G e H_T para código Hamming (7,4)
    G_74 = np.array([
        [1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1]
    ])
    
    H_T_74 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Probabilidades de erro do canal
    erro_canal = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 
                 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002]
    
    # Número de bits para simulação
    num_bits = 10000000
    
    print("Simulando código Hamming (9,5)...")
    resultados_95 = simular_transmissao(G_95, H_T_95, erro_canal, num_bits)
    
    print("\nSimulando código Hamming (7,4)...")
    resultados_74 = simular_transmissao(G_74, H_T_74, erro_canal, num_bits)
    
    print("\nCalculando caso sem codificação...")
    resultados_sem_codigo = simular_sem_codigo(erro_canal, num_bits)
    
    # Plotar gráfico comparativo
    plotar_comparacao(resultados_95, resultados_74, resultados_sem_codigo)
    
    return resultados_95, resultados_74, resultados_sem_codigo

# Executa o exemplo
if __name__ == "__main__":
    funcao_final()
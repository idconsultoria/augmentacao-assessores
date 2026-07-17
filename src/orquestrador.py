"""
Orquestrador da pipeline de Aumentação de Assessores.

Fluxo principal:
  1. Conecta ao Google Drive (via Agemini)
  2. Busca relatórios PDF pendentes
  3. Identifica clientes via Google Sheets (PROCV)
  4. Valida relatórios com Gemini (carteira ativa?)
  5. Gera mensagem personalizada com Gemini (N-Shot + Structured Output)
  6. Envia via Baileys (PDF + texto)
  7. Registra logs no Google Sheets

Dependências:
  - agemini.conectores (Google Drive, Sheets)
  - agemini.modelos (Gemini)
  - src.whatsapp (BaileysClient)
  - src.config (AssessorConfig)
"""

import os
import io
import re
import sys
import json
import base64
import zipfile
import shutil
import tempfile
from datetime import datetime

import requests
import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from .config import AssessorConfig
from .whatsapp import BaileysClient


# ==========================================
# GOOGLE DRIVE & SHEETS (via Agemini)
# ==========================================

def obter_servicos_google(config: AssessorConfig):
    """
    Autentica nos serviços Google (Drive + Sheets).
    Prioridade: env var JSON > arquivo local > ADC (Cloud Run service account).
    """
    escopos = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets',
    ]

    credenciais_json = config.google_credentials_json
    if credenciais_json:
        print("[*] Credenciais Google via variável de ambiente (modo Cloud).")
        info = json.loads(credenciais_json)
        credenciais = service_account.Credentials.from_service_account_info(info, scopes=escopos)
    else:
        info = config.get_google_credentials()
        if info:
            print("[*] Credenciais Google via arquivo JSON (modo Dev).")
            credenciais = service_account.Credentials.from_service_account_info(info, scopes=escopos)
        else:
            # Application Default Credentials (Cloud Run service account)
            print("[*] Credenciais Google via ADC (Cloud Run).")
            import google.auth
            credenciais, _proj = google.auth.default(scopes=escopos)

    drive_service = build('drive', 'v3', credentials=credenciais)
    sheets_service = build('sheets', 'v4', credentials=credenciais)
    return drive_service, sheets_service


def buscar_relatorios_pendentes(drive_service, id_pasta: str) -> list:
    """Busca PDFs na pasta de pendentes do Google Drive."""
    print(f"[*] Buscando relatórios PDF na pasta de pendentes...")
    query = f"'{id_pasta}' in parents and mimeType='application/pdf' and trashed=false"
    resultados = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return resultados.get('files', [])


def extrair_pdfs_de_zips(drive_service, id_pasta_pendentes: str, id_pasta_processados: str) -> list:
    """
    Busca ZIPs, extrai PDFs localmente e faz upload para a pasta de pendentes.
    Após sucesso, move o ZIP para processados.
    Retorna lista de dicts {'id', 'name'} dos PDFs extraídos.
    """
    print(f"[*] Verificando arquivos ZIP na pasta de pendentes...")
    query = (
        f"'{id_pasta_pendentes}' in parents and trashed=false and ("
        f"mimeType='application/zip' or "
        f"mimeType='application/x-zip-compressed' or "
        f"mimeType='application/x-zip' or "
        f"mimeType='application/octet-stream' or "
        f"name contains '.zip'"
        f")"
    )
    resultados = drive_service.files().list(q=query, fields="files(id, name)").execute()
    zips_encontrados = resultados.get('files', [])

    if not zips_encontrados:
        print(f"    [*] Nenhum arquivo ZIP encontrado. Prosseguindo.")
        return []

    print(f"    [+] {len(zips_encontrados)} arquivo(s) ZIP encontrado(s).")
    pdfs_subidos = []

    for zip_info in zips_encontrados:
        zip_id = zip_info['id']
        zip_nome = zip_info['name']
        caminho_zip_local = os.path.join(tempfile.gettempdir(), zip_nome)
        pasta_extracao = os.path.join(tempfile.gettempdir(), f"_extracao_{zip_nome.replace('.zip', '')}")

        pdfs_deste_zip = []
        todos_subidos = False

        try:
            # Baixar ZIP
            print(f"        [*] Baixando ZIP: {zip_nome}...")
            request = drive_service.files().get_media(fileId=zip_id)
            with io.FileIO(caminho_zip_local, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            # Extrair
            os.makedirs(pasta_extracao, exist_ok=True)
            with zipfile.ZipFile(caminho_zip_local, 'r') as zip_ref:
                zip_ref.extractall(pasta_extracao)

            pdfs_na_pasta = [f for f in os.listdir(pasta_extracao) if f.lower().endswith('.pdf')]

            if not pdfs_na_pasta:
                print(f"        [!] Nenhum PDF dentro do ZIP '{zip_nome}'.")
                todos_subidos = True
            else:
                sucessos = 0
                for pdf_nome in pdfs_na_pasta:
                    caminho_pdf_local = os.path.join(pasta_extracao, pdf_nome)
                    try:
                        file_metadata = {'name': pdf_nome, 'parents': [id_pasta_pendentes]}
                        media = MediaFileUpload(caminho_pdf_local, mimetype='application/pdf', resumable=True)
                        arquivo_criado = drive_service.files().create(
                            body=file_metadata, media_body=media, fields='id, name'
                        ).execute()
                        print(f"        [+] Upload: {pdf_nome} (ID: {arquivo_criado['id']})")
                        pdfs_deste_zip.append(arquivo_criado)
                        pdfs_subidos.append(arquivo_criado)
                        sucessos += 1
                    except Exception as e:
                        print(f"        [-] Falha upload '{pdf_nome}': {e}")
                    finally:
                        if os.path.exists(caminho_pdf_local):
                            os.remove(caminho_pdf_local)

                todos_subidos = (sucessos == len(pdfs_na_pasta))

        except Exception as e:
            print(f"        [-] Erro ao processar ZIP '{zip_nome}': {e}")
        finally:
            if os.path.exists(caminho_zip_local):
                os.remove(caminho_zip_local)
            if os.path.exists(pasta_extracao):
                shutil.rmtree(pasta_extracao, ignore_errors=True)

        if todos_subidos and pdfs_deste_zip:
            mover_arquivo_drive(drive_service, zip_id, id_pasta_processados)
            print(f"        [+] ZIP '{zip_nome}' movido para Processados.")
        elif not todos_subidos:
            print(f"        [!] ZIP '{zip_nome}' NÃO movido (nem todos PDFs subidos).")

    print(f"\n[+] Processamento de ZIPs concluído. {len(pdfs_subidos)} PDF(s) extraídos.")
    return pdfs_subidos


def descarregar_relatorio(drive_service, file_id: str, nome_destino: str) -> str:
    """Baixa um PDF do Drive para /tmp e retorna o caminho local."""
    caminho_local = os.path.join(tempfile.gettempdir(), nome_destino)
    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(caminho_local, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return caminho_local


def mover_arquivo_drive(drive_service, file_id: str, id_pasta_destino: str):
    """Move um arquivo do Drive para outra pasta."""
    arquivo = drive_service.files().get(fileId=file_id, fields='parents').execute()
    parents_atuais = ",".join(arquivo.get('parents', []))
    drive_service.files().update(
        fileId=file_id,
        addParents=id_pasta_destino,
        removeParents=parents_atuais,
        fields='id, parents'
    ).execute()


def ler_personalizacao_assessor(drive_service, doc_id: str) -> str:
    """Lê o Google Doc de personalização do assessor."""
    if not doc_id:
        return ""
    print(f"[*] Lendo manual de personalização do assessor...")
    try:
        request = drive_service.files().export_media(fileId=doc_id, mimeType='text/plain')
        return request.execute().decode('utf-8')
    except Exception as e:
        print(f"[-] Erro ao ler documento de personalização: {e}")
        return ""


# ==========================================
# GOOGLE SHEETS (PROCV e Logs)
# ==========================================

def formatar_telefone(telefone_bruto: str) -> str:
    """
    Limpa e formata telefone para padrão internacional.
    Números brasileiros (10-11 dígitos) recebem DDI 55.
    """
    if not telefone_bruto:
        return ""
    apenas_numeros = re.sub(r'\D', '', str(telefone_bruto))
    if not apenas_numeros:
        return ""
    if len(apenas_numeros) in [10, 11] and not apenas_numeros.startswith("55"):
        return f"55{apenas_numeros}"
    return apenas_numeros


def obter_sheet_id(sheets_service, id_planilha: str, nome_aba: str) -> int:
    """Obtém o sheetId de uma aba pelo nome."""
    planilha = sheets_service.spreadsheets().get(spreadsheetId=id_planilha).execute()
    for sheet in planilha.get('sheets', []):
        if sheet['properties']['title'] == nome_aba:
            return sheet['properties']['sheetId']
    return None


def buscar_dados_cliente(
    sheets_service,
    id_planilha: str,
    nome_aba_clientes: str,
    identificador_relatorio: str,
) -> dict | None:
    """
    Simula PROCV: busca cliente na planilha pelo identificador no nome do arquivo.
    Colunas: A:ID | B:Status | C:Nome | D:Telefone | E:Vocativo |
             F:Tom de Voz | G:Instruções | H:Status Envio | I:Ultimo Resumo
    """
    intervalo = f"'{nome_aba_clientes}'!A2:I10000"
    resultado = sheets_service.spreadsheets().values().get(
        spreadsheetId=id_planilha, range=intervalo
    ).execute()

    linhas = resultado.get('values', [])
    for idx, linha in enumerate(linhas):
        if len(linha) > 0 and linha[0] in identificador_relatorio:
            numero_linha = idx + 2
            telefone_bruto = linha[3] if len(linha) > 3 else ""
            telefone_formatado = formatar_telefone(telefone_bruto)

            return {
                "identificador": linha[0],
                "status": linha[1] if len(linha) > 1 else "Envio Ativo",
                "nome": linha[2] if len(linha) > 2 else "",
                "telefone": telefone_formatado,
                "vocativo": linha[4] if len(linha) > 4 else "",
                "tom_de_voz": linha[5] if len(linha) > 5 else "formal e objetivo",
                "instrucoes_assessor": linha[6] if len(linha) > 6 else "Nenhuma instrução adicional.",
                "ultimo_resumo": linha[8] if len(linha) > 8 else "Sem histórico",
                "numero_linha": numero_linha,
            }
    return None


def registrar_log(
    sheets_service,
    id_planilha: str,
    nome_aba_logs: str,
    nome_aba_clientes: str,
    id_relatorio: str,
    nome_cliente: str,
    status: str,
    detalhes: str,
):
    """Registra log na aba de registros, inserindo na linha 2 (mais recente primeiro)."""
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_checkbox = status == "SUCESSO"

    sheet_id = obter_sheet_id(sheets_service, id_planilha, nome_aba_logs)
    if sheet_id is not None:
        request_body = {
            'requests': [{
                'insertDimension': {
                    'range': {
                        'sheetId': sheet_id,
                        'dimension': 'ROWS',
                        'startIndex': 1,
                        'endIndex': 2
                    },
                    'inheritFromBefore': False
                }
            }]
        }
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=id_planilha, body=request_body
        ).execute()

    sheets_service.spreadsheets().values().update(
        spreadsheetId=id_planilha,
        range=f"'{nome_aba_logs}'!A2:E2",
        valueInputOption="USER_ENTERED",
        body={'values': [[agora, id_relatorio, nome_cliente, status_checkbox, detalhes]]}
    ).execute()
    print(f"[+] Log registrado: {status}")


def marcar_status_envio(sheets_service, id_planilha: str, nome_aba: str, numero_linha: int, sucesso: bool):
    """Marca checkbox de status de envio na aba Clientes."""
    sheets_service.spreadsheets().values().update(
        spreadsheetId=id_planilha,
        range=f"'{nome_aba}'!H{numero_linha}",
        valueInputOption="USER_ENTERED",
        body={'values': [[sucesso]]}
    ).execute()


def atualizar_ultimo_resumo(sheets_service, id_planilha: str, nome_aba: str, numero_linha: int, mensagem: str):
    """Atualiza coluna de último resumo na aba Clientes."""
    sheets_service.spreadsheets().values().update(
        spreadsheetId=id_planilha,
        range=f"'{nome_aba}'!I{numero_linha}",
        valueInputOption="USER_ENTERED",
        body={'values': [[mensagem]]}
    ).execute()


# ==========================================
# GEMINI — Validação e Geração
# ==========================================

def validar_relatorio_ativo(caminho_pdf: str, config: AssessorConfig) -> dict:
    """
    Usa Gemini para validar se o relatório contém carteira ativa.
    Retorna {"valido": bool, "justificativa": str}.
    """
    print("    [*] Validando se o relatório contém dados ativos...")
    genai.configure(api_key=config.gemini_api_key)
    ficheiro = genai.upload_file(caminho_pdf)

    system_instruction = """
    Você é um auditor financeiro rigoroso. Sua única função é analisar este relatório de performance de investimentos (PDF) e determinar se a carteira possui movimentação real ou se está "fantasma/zerada".

    Uma carteira DEVE ser considerada INATIVA/INVÁLIDA (valido: false) se você observar:
    - "Rentabilidade Mês" de 0,00% E "Ganho Mês" de R$ 0,00.
    - Gráficos de "Evolução Patrimonial" ou "Rentabilidade" completamente retos (linhas no zero).
    - Tabelas de referência com 0,00% em todos os meses preenchidos.
    - Patrimônio travado em um valor X mas sem nenhuma oscilação, ganho financeiro ou rentabilidade histórica.

    Responda EXATAMENTE com um objeto JSON contendo:
    - "valido": booleano (true se houver oscilação/rendimento real, false se for inativa/zerada).
    - "justificativa": string curta de até 10 palavras explicando o motivo da decisão.
    """

    generation_config = genai.GenerationConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "valido": {"type": "BOOLEAN"},
                "justificativa": {"type": "STRING"},
            },
            "required": ["valido", "justificativa"],
        }
    )

    modelo = genai.GenerativeModel(
        model_name=config.gemini_model,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    try:
        resposta = modelo.generate_content([
            ficheiro,
            "Analise os indicadores de rentabilidade e ganho deste relatório e retorne o JSON de validação."
        ])
        return json.loads(resposta.text)
    except Exception as e:
        print(f"    [-] Erro na validação: {e}")
        return {"valido": True, "justificativa": "Falha na validação prévia."}


def gerar_mensagem_assessor(
    caminho_pdf: str,
    config: AssessorConfig,
    dados_cliente: dict,
    ultimo_resumo: str,
    personalizacao_assessor: str,
) -> tuple[str, str]:
    """
    Gera mensagem personalizada de WhatsApp usando Gemini.
    Usa N-Shot (exemplos) + Structured Outputs (JSON).

    Returns:
        (mensagem_whatsapp, resumo_interno)
    """
    api_key = config.gemini_api_key
    genai.configure(api_key=api_key)
    ficheiro = genai.upload_file(caminho_pdf)

    tom_especifico = dados_cliente.get("tom_de_voz", "profissional e direto")
    vocativo = dados_cliente.get("vocativo", "").strip() or dados_cliente.get("nome", "Cliente")
    instrucoes = dados_cliente.get("instrucoes_assessor", "Nenhuma instrução adicional.")
    contexto = ultimo_resumo if ultimo_resumo else "Este é o primeiro relatório do cliente."

    system_instruction = f"""
# PAPEL E OBJETIVO
Você é um Assessor de Investimentos atuando em nome de um escritório (afiliado à XP Investimentos). Sua única responsabilidade é analisar relatórios mensais de performance de carteira em PDF (XPerformance) e redigir uma mensagem de WhatsApp para o cliente cujo vocativo é {vocativo}. O objetivo é resumir o rendimento do mês/ano, destacar os principais pontos e se colocar à disposição, enviando o PDF em anexo.

# PERSONALIZAÇÃO PARA ESTE CLIENTE
- Tom de voz: {tom_especifico}
- Instruções do Assessor: {instrucoes}
- Contexto anterior: "{contexto}"

# TOM DE VOZ E PERSONALIDADE (GERAL)
- Próximo e Empático: Trate o cliente pelo nome.
- Consultivo e Transparente: Comemore bons resultados, mas não esconda resultados ruins.
- Profissional mas Conversacional: Evite jargões sem explicação. Use linguagem de WhatsApp com BOA gramática.
- Propositivo: Traga visão de futuro quando fizer sentido.

# FORMATAÇÃO WHATSAPP
- *negrito* para números e indicadores importantes
- _itálico_ para nomes de relatórios
- Emojis com moderação (máx 3-4): 📈 📊 🤝 💡 👋
- Parágrafos curtos

# REGRAS
1. NUNCA prometa rentabilidade futura.
2. Baseie-se ESTRITAMENTE nos dados do PDF.
3. A mensagem pressupõe que o PDF será enviado em anexo.
4. Retorne EXATAMENTE um JSON com "mensagem_whatsapp" e "resumo_interno".

# PERSONALIZAÇÃO DO ASSESSOR
{personalizacao_assessor}
"""

    generation_config = genai.GenerationConfig(
        temperature=1,
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "mensagem_whatsapp": {"type": "STRING"},
                "resumo_interno": {"type": "STRING"},
            },
            "required": ["mensagem_whatsapp", "resumo_interno"],
        }
    )

    # N-Shot examples
    historico = [
        {
            "role": "user",
            "parts": ["Analise o relatório (Perfil: Cliente geral, resultado focado no CDI, Bom resultado)."]
        },
        {
            "role": "model",
            "parts": [json.dumps({
                "mensagem_whatsapp": "Olá Breno, bom dia! 👋\n\nSegue o seu _relatório de performance_ atualizado até o fim deste mês.\n\nSua carteira está indo muito bem! Desde o início da nossa alocação, já acumulamos um rendimento de *122% do CDI*. Tá show! 📈\n\nDá uma olhada no PDF e, se quiser bater um papo para revisarmos os próximos passos, é só me avisar. Abraço!",
                "resumo_interno": "Resultado focado no CDI. Bom resultado de 122%."
            }, ensure_ascii=False)]
        },
        {
            "role": "user",
            "parts": ["Analise o relatório (Perfil: Cliente sênior/formal, foco em segurança)."]
        },
        {
            "role": "model",
            "parts": [json.dumps({
                "mensagem_whatsapp": "Bom dia, Dona Rosa. Como a senhora está? 🌻\n\nEstou enviando o relatório de performance da sua carteira. O ano está indo muito bem, já acumulamos *10% de rentabilidade* (o que equivale a *121% do CDI*). É um resultado excelente e bem seguro.\n\nQualquer dúvida, estou à disposição.",
                "resumo_interno": "Segurança e clareza. Rentabilidade de 10% (121% CDI)."
            }, ensure_ascii=False)]
        },
        {
            "role": "user",
            "parts": ["Analise o relatório (Perfil: Ações bem, Renda Fixa mal, Visão macro)."]
        },
        {
            "role": "model",
            "parts": [json.dumps({
                "mensagem_whatsapp": "Fala Rafa, boa tarde! 🤝\n\nSegue o fechamento da sua carteira. A parte de *Renda Fixa - Inflação* acabou sendo a pior performance do mês, mas, em compensação, a parte de *Renda Variável* segurou muito bem e puxou o resultado para cima. 📊\n\nDe todo modo, eu não mudaria nada agora. Já já os juros começam a cair e essa parte atrelada à inflação volta a valorizar também. Qualquer dúvida, me chama aqui!",
                "resumo_interno": "Ações bem, Renda Fixa mal. Expectativa de queda de juros."
            }, ensure_ascii=False)]
        },
        {
            "role": "user",
            "parts": ["Analise o relatório (Perfil: Foco em Dividendos)."]
        },
        {
            "role": "model",
            "parts": [json.dumps({
                "mensagem_whatsapp": "Fala Igor, tudo bem? Bom dia!\n\nSegue o seu _XPerformance_ fresquinho. O grande destaque desse mês continua sendo a nossa *carteira top dividendos*, que está indo muito bem e gerando uma ótima renda passiva. 💸\n\nO rendimento global fechou em *110% do CDI*. Dá uma olhada no anexo e me diz o que achou.",
                "resumo_interno": "Foco em Dividendos. Rendimento global 110% do CDI."
            }, ensure_ascii=False)]
        },
    ]

    modelo = genai.GenerativeModel(
        model_name=config.gemini_model,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    print("[*] Iniciando chat com exemplos N-Shot...")
    chat = modelo.start_chat(history=historico)

    print("[*] Analisando o relatório atual...")
    try:
        resposta = chat.send_message([
            ficheiro,
            "Aqui está o relatório atual do cliente. Gere a síntese seguindo as diretrizes e o formato exigido."
        ])
        resultado = json.loads(resposta.text)
        return resultado["mensagem_whatsapp"], resultado["resumo_interno"]
    except Exception as e:
        # Fallback: usa template N-Shot deterministico
        print(f"[-] Erro no Gemini: {e}")
        print("    [*] Usando fallback deterministico (N-Shot)...")
        return _gerar_mensagem_fallback(dados_cliente)


def _gerar_mensagem_fallback(dados_cliente: dict) -> tuple[str, str]:
    """Fallback deterministico: seleciona template N-Shot por tom de voz."""
    nome = dados_cliente.get("nome", "Cliente")
    vocativo = dados_cliente.get("vocativo", "").strip() or nome
    tom = dados_cliente.get("tom_de_voz", "").lower()

    # Templates por perfil de tom
    if "formal" in tom or "senior" in tom or "sênior" in tom:
        msg = f"Bom dia, {vocativo}! 🌻\n\nEstou enviando o relatório de performance da sua carteira. O ano está indo muito bem.\n\nQualquer dúvida, estou à disposição. Abraço!"
        resumo = "Formal/sênior."
    elif "dividendo" in tom:
        msg = f"Olá {vocativo}! 💸\n\nSegue o seu _XPerformance_. O destaque continua sendo a nossa carteira de dividendos.\n\nDá uma olhada no anexo!"
        resumo = "Foco em dividendos."
    elif "empatico" in tom or "próximo" in tom:
        msg = f"Olá {vocativo}, bom dia! 👋\n\nSegue o seu _relatório de performance_. Sua carteira está indo bem!\n\nDá uma olhada no PDF e qualquer dúvida é só me chamar. Abraço!"
        resumo = "Empático e próximo."
    else:
        msg = f"Olá {vocativo}! 🤝\n\nSegue o relatório da sua carteira.\n\nO mercado está ajudando e nosso posicionamento está bem aderente. Qualquer dúvida, estou por aqui!"
        resumo = "Padrão."

    return msg, f"[Fallback] {resumo}"


# ==========================================
# FLUXO PRINCIPAL
# ==========================================

def executar(config: AssessorConfig, whatsapp_client=None):
    """
    Pipeline principal para um assessor.

    Args:
        config: Configuração do assessor (AssessorConfig)
        whatsapp_client: BaileysClient opcional. Se omitido, cria um novo.
    """
    print(f"\n{'='*60}")
    print(f"  🚀 AUMENTAÇÃO DE ASSESSORES — {config.nome}")
    print(f"{'='*60}\n")

    # ── Conexões ──────────────────────────────────────────
    drive_service, sheets_service = obter_servicos_google(config)
    whatsapp = whatsapp_client or BaileysClient(base_url=config.whatsapp_service_url)

    # ── ZIPs → PDFs ───────────────────────────────────────
    extrair_pdfs_de_zips(drive_service, config.id_pasta_pendentes, config.id_pasta_processados)

    # ── Personalização do assessor ────────────────────────
    personalizacao = ler_personalizacao_assessor(drive_service, config.id_doc_personalizacao)

    # ── Buscar PDFs pendentes ────────────────────────────
    relatorios = buscar_relatorios_pendentes(drive_service, config.id_pasta_pendentes)
    itens = [{'nome': r['name'], 'drive_id': r['id']} for r in relatorios]

    if not itens:
        print("Nenhum relatório PDF encontrado. Encerrando.")
        return

    limite = config.envios_por_execucao
    print(f"\n[*] {len(itens)} relatório(s) a processar. Limite: {limite}.\n")
    processados = 0

    for item in itens:
        if processados >= limite:
            print(f"\n[!] Limite de {limite} envios atingido. Encerrando.")
            break

        nome_arquivo = item['nome']
        drive_id = item['drive_id']
        print(f"\n--- Processando ({processados + 1}/{limite}): {nome_arquivo} ---")

        # 1. Buscar cliente (PROCV)
        dados_cliente = buscar_dados_cliente(
            sheets_service, config.id_planilha_clientes,
            config.nome_aba_clientes, nome_arquivo
        )

        if not dados_cliente:
            print(f"[-] Cliente não encontrado para {nome_arquivo}.")
            registrar_log(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_logs, config.nome_aba_clientes,
                nome_arquivo, "Desconhecido", "ERRO", "Cliente não encontrado"
            )
            continue

        print(f"[+] Cliente: {dados_cliente['nome']} | Status: {dados_cliente['status']}")

        # Verificar status de pausa
        if dados_cliente['status'] == "Envio Pausado":
            print(f"    [!] Envio pausado. Movendo arquivo e ignorando...")
            mover_arquivo_drive(drive_service, drive_id, config.id_pasta_processados)
            registrar_log(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_logs, config.nome_aba_clientes,
                dados_cliente['identificador'], dados_cliente['nome'],
                "IGNORADO", "Envio Pausado"
            )
            continue

        # 2. Baixar PDF
        print(f"[*] Baixando PDF do Drive...")
        caminho_local = descarregar_relatorio(drive_service, drive_id, nome_arquivo)

        # 3. Validar relatório
        validacao = validar_relatorio_ativo(caminho_local, config)

        if not validacao.get("valido", True):
            motivo = validacao.get('justificativa', 'Carteira zerada/inativa')
            print(f"    [!] Relatório inválido: {motivo}")
            print(f"    [*] Pulando envio e movendo para processados...")
            mover_arquivo_drive(drive_service, drive_id, config.id_pasta_processados)
            registrar_log(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_logs, config.nome_aba_clientes,
                dados_cliente['identificador'], dados_cliente['nome'],
                "PULADO", motivo
            )
            if os.path.exists(caminho_local):
                os.remove(caminho_local)
            continue

        # 4. Gerar mensagem com IA
        print("[*] Gerando mensagem com IA...")
        msg_whatsapp, resumo_log = gerar_mensagem_assessor(
            caminho_local, config,
            dados_cliente,
            dados_cliente.get('ultimo_resumo', 'Sem histórico'),
            personalizacao,
        )

        # 5. Enviar via Baileys
        sucesso = whatsapp.send_pdf_and_text(
            dados_cliente['telefone'], caminho_local, nome_arquivo, msg_whatsapp
        )

        # 6. Pós-processamento
        if sucesso:
            processados += 1
            print(f"[+] Envio bem-sucedido! ({processados}/{limite} hoje)")
            mover_arquivo_drive(drive_service, drive_id, config.id_pasta_processados)
            registrar_log(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_logs, config.nome_aba_clientes,
                dados_cliente['identificador'], dados_cliente['nome'],
                "SUCESSO", msg_whatsapp
            )
            atualizar_ultimo_resumo(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_clientes,
                dados_cliente['numero_linha'], msg_whatsapp
            )
            marcar_status_envio(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_clientes,
                dados_cliente['numero_linha'], True
            )
        else:
            print("[-] Falha no envio.")
            registrar_log(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_logs, config.nome_aba_clientes,
                dados_cliente['identificador'], dados_cliente['nome'],
                "FALHA_ENVIO", "Erro no serviço Baileys"
            )
            marcar_status_envio(
                sheets_service, config.id_planilha_clientes,
                config.nome_aba_clientes,
                dados_cliente['numero_linha'], False
            )

        # Limpeza
        if os.path.exists(caminho_local):
            os.remove(caminho_local)

    print(f"\n[✓] Pipeline concluída para {config.nome}. "
          f"Envios bem-sucedidos: {processados}/{limite}.")

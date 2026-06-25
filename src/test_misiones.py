import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mission_manager import get_available_missions, get_mission_by_id, build_beginner_prompt
from chat_ollama import parse_response

def test_database():
    print("[1] Probando carga de misiones...")
    missions = get_available_missions()
    assert len(missions) > 0, "No se cargo ninguna categoria"
    print(f"   -> Categorias cargadas: {list(missions.keys())}")
    
    # Verificar la presencia de las misiones basicas
    all_basic_ids = ["how_much_is_it", "find_the_product", "understand_the_menu", "order_food", 
                     "organize_morning", "find_the_gate", "first_day_work", "describe_symptom"]
    
    found_count = 0
    for id_ in all_basic_ids:
        mission = get_mission_by_id(id_)
        if mission:
            found_count += 1
            print(f"   [OK] Mision cargada: {mission['title']}")
        else:
            print(f"   [FALLO] No se encontro la mision con ID: {id_}")
            
    assert found_count == len(all_basic_ids), "No se cargaron todas las misiones basicas de forma correcta"
    print("-> Carga de base de datos exitosa.")

def test_prompt_builder():
    print("\n[2] Probando el generador de prompts para principiantes...")
    mission = get_mission_by_id("how_much_is_it")
    prompt = build_beginner_prompt(mission)
    
    assert "strict roleplay" in prompt.lower(), f"Falto 'strict roleplay' en el prompt. Prompt: {prompt}"
    assert "friendly but cost-conscious tourist" in prompt.lower(), f"Falto el rol de Neobit en el prompt. Prompt: {prompt}"
    assert "a1 level" in prompt.lower(), f"Falto 'A1 level' en el prompt. Prompt: {prompt}"
    assert "[hablar]" in prompt.lower(), f"Falto '[HABLAR]' en el prompt. Prompt: {prompt}"
    assert "[nota]" in prompt.lower(), f"Falto '[NOTA]' en el prompt. Prompt: {prompt}"
    print("   -> Prompt de prueba generado exitosamente y validado.")
    print("-> Generador de prompts funciona correctamente.")

def test_parsing():
    print("\n[3] Probando el parser de respuesta dual...")
    response_raw = """
[HABLAR]
Hello! How much is this key chain? Is it cheap?
[/HABLAR]
[NOTA]
Gramatica: Excelente trabajo
Pronunciacion: Souvenir [su-ve-nir]
[/NOTA]
"""
    parsed = parse_response(response_raw)
    assert parsed["spoken"] == "Hello! How much is this key chain? Is it cheap?", f"Conversacion fallo: {parsed['spoken']}"
    assert "Gramatica:" in parsed["note"], f"Nota fallo: {parsed['note']}"
    print(f"   -> Spoken extraido: \"{parsed['spoken']}\"")
    print(f"   -> Note extraido: \"{parsed['note']}\"")
    print("-> Parser de respuesta funciona correctamente.")

if __name__ == "__main__":
    print("=== INICIANDO PRUEBAS DE INTEGRACION DE NEOBIT MISSION SYSTEM ===")
    try:
        test_database()
        test_prompt_builder()
        test_parsing()
        print("\n=== TODAS LAS PRUEBAS UNITARIAS PASARON EXITOSAMENTE! ===")
    except AssertionError as e:
        print(f"\nERROR DE ASERCION: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR INESPERADO: {e}")
        sys.exit(1)

import sys
import os
import asyncio

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from vinculacion.vinculacion_main import VinculacionOrchestrator
from vinculacion._linking_algorithm import UOFUPLinkingAlgorithm
from vinculacion._vinculacion_monitoring import UPChangeMonitor


async def linking_algorithm_test():

    # Initialize the algorithm
    """
    Test the UOF to UP linking algorithm by attempting to link specified UPs for a target date.
    
    Runs the linking process for the UPs "ZABU" and "TERE" using the configured target date. Prints and saves the resulting links to a CSV file if successful; otherwise, prints an error message.
    """
    algorithm = UOFUPLinkingAlgorithm()
    # Get the target date
    target_date = algorithm.config.get_linking_target_date()
    ups_to_link = ["ZABU", "TERE"]  # If None, all active UPs will be linked
    results = await algorithm.link_uofs_to_ups(target_date = target_date, ups_to_link=ups_to_link)
    
    if results['success']:
        links_df = results['links_df']
        print(links_df)
        links_df.to_csv('links_df.csv', index=False)
    else:
        print(f"Linking process failed: {results['message']}")

async def change_monitor_test():
    """
    Run a test to monitor changes in existing organizational links using the UPChangeMonitor.
    """
    change_monitor = UPChangeMonitor()
    await change_monitor.monitor_existing_links()

async def full_linking_test(initial_linking: bool = False):
    """
    Runs the full Vinculacion linking workflow, optionally performing initial linking and always monitoring for changes.
    
    Parameters:
        initial_linking (bool): If True, performs a full initial linking of UOFs to UPs before monitoring.
    
    This function orchestrates the main Vinculacion process: it can create initial links between organizational units and positions, then monitors for changes or de-linking in existing links. Progress and completion messages are printed for clarity.
    """
    orchestrator = VinculacionOrchestrator()
    
    print("="*100)
    print("🎯 VINCULACION MODULE - UOF TO UP LINKING")
    print("🗓️  Target Date: Automatically set to 93 days back")
    print("="*100)

    # === STEP 0: FULL LINKING ===
    # This should run first to create the initial links.
    if initial_linking:
        await orchestrator.perform_full_linking()

    # === STEP 1: MONITOR EXISTING LINKS FOR CHANGES ===
    # This should run first to detect any de-linking or changes before adding new ones.
    await orchestrator.perform_vinculacion_monitoring()

    # === STEP 2: INCREMENTAL CHECK FOR NEWLY ENABLED UPs ===
    # This checks for UPs that became active 93 days ago and links them if they aren't already.
    #await orchestrator.perform_new_ups_linking()
    #breakpoint()

    print("\n" + "="*100)
    print("🏁 VINCULACION PROCESS COMPLETE")
    print("="*100)

if __name__ == "__main__":

    # asyncio.run(linking_algorithm_test())   
    # asyncio.run(change_monitor_test())
    asyncio.run(full_linking_test(initial_linking=False)) 


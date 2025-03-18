### 05/28/2024
### Using Inotify to monitor /hdd/data/voltages/
### when a new .nc file is created, it calls the T3 plotting task, saves a png/pdf file, and posts it to the Slack candidates channel.
### Usage: poetry run python T3_monitor.py

import inotify.adapters as ia
import os
import sys
import slack_sdk as slk
import time
import logging
import cand_plotter

logfile = '/home/cugrex/grex/t3/services/T3_plotter.log'
env_dir = "/home/cugrex/grex/t3/grex_t3/"
mon_dir = "/hdd/data/voltages/" # monitoring dir                                                 
dir_plot = "/hdd/data/candidates/T3/candplots/" # place to save output plots                     
dir_fil  = "/hdd/data/candidates/T3/candfils/"

# Create directories if they don't exist
for directory in [mon_dir, dir_plot, dir_fil]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

# Configure the logger
logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Starting the monitoring and T3-plotting service')


# Function to upload a plot to Slack
def upload_to_slack(pdffile):

    # set up slack client
    slack_file = '{0}/.config/slack_api'.format(
        os.path.expanduser("~")
    )

    # if the slack api token file is missing
    if not os.path.exists(slack_file):
        raise RuntimeError(
            "Could not find file with slack api token at {0}".format(
                slack_file
            )
        )
    # otherwise load the token file and start a webclient talking to slack
    with open(slack_file) as sf_handler:
        slack_token = sf_handler.read()
        # initialize slack client
        client = slk.WebClient(token=slack_token)
    
    # Define message parameters
    message = "New candidate plot generated!" ## add some cand details?
    
    try:
        # Upload the plot file to Slack
        response = client.files_upload_v2(
            channel="C07M2900B7W",
            file=pdffile,
            initial_comment=message
        )
        
        print("Plot uploaded to Slack:", response["file"]["permalink"])
    except slk.errors.SlackApiError as err:
        print(f"Error uploading plot to Slack: {err}")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise 

# Function to send a slack message (test)
def send_to_slack(message):

    # set up slack client
    slack_file = '{0}/.config/slack_api'.format(
        os.path.expanduser("~")
    )

    # if the slack api token file is missing
    if not os.path.exists(slack_file):
        raise RuntimeError(
            "Could not find file with slack api token at {0}".format(
                slack_file
            )
        )
    # otherwise load the token file and start a webclient talking to slack
    with open(slack_file) as sf_handler:
        slack_token = sf_handler.read()
        # initialize slack client
        client = slk.WebClient(token=slack_token)
    
    # Define message parameters
    try:
        response = client.chat_postMessage(
            channel="candidates-cuithaca",
            text=message
        )  
        
        print("Done", response.status_code)
    except slk.errors.SlackApiError as err:
        print(f"Error uploading plot to Slack: {err}")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

def process_candidate(filename, c):
    """Process a single candidate file and return the plot path"""
    v = mon_dir + f"grex_dump-{c}.nc"
    fn_tempfil = dir_plot + "intermediate.fil"
    fn_outfil = dir_fil + f"cand{c}.fil"
    
    # Generate and plot candidate
    cand, tab = cand_plotter.gen_cand(v, fn_tempfil, fn_outfil, f'{c}.json')
    cand_plotter.plot_grex(cand, tab, f"{c}.json")
    logging.info("Done with cand_plotter.py")
    
    # Cleanup temporary file
    os.remove(fn_tempfil)
    logging.info("Successfully plotted the candidate!")
    
    return dir_plot + f"grex_cand{c}.png"

def main(path, post=True):
    print('Starting the monitoring and T3-plotting service')
    #try:
    # Initialize monitoring
    i = ia.Inotify()
    i.add_watch(path)
    
    # Create monitor start marker
    monitor_file = path + 'start_inotify_monitor'
    with open(monitor_file, 'w') as f:
        pass
    
    # Main monitoring loop
    for event in i.event_gen(yield_nones=False):
        _, type_names, path, filename = event
        
        print('event:', event)

        # Demand that the event is a file creation event, not just altered.
        if type_names != ['IN_CREATE'] or not filename.endswith('.nc'):
            continue
            
        logging.info(f"New NetCDF file created: {filename}")
        
        # Extract candidate ID and prepare
        c = filename.split('.')[0].split('/')[-1].split('-')[-1]
        os.chdir(env_dir)
        print('Sleeping for 10 seconds')
        time.sleep(10)  # Wait for file to be fully written
        
        try:
            # Process candidate and get plot path
            plot_path = process_candidate(filename, c)
            print('plot_path:', plot_path)
            # Post to Slack if requested
            if post:
                upload_to_slack(plot_path)
                logging.info("Successfully posted to Slack #candidates!")
            
        except FileNotFoundError as e:
            logging.error(f"Required file not found: {str(e)}")
        except slk.errors.SlackApiError as e:
            logging.error(f"Slack API error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error processing candidate {c}: {str(e)}")
                
    # except PermissionError:
    #     logging.error("Permission denied: Unable to create inotify test file.")
    # except Exception as e:
    #     logging.error(f"Fatal error in monitoring: {str(e)}")


if __name__ == '__main__':
    try:
        post = True
        main(mon_dir, post=post)
    except Exception as e:
        print('Interrupted')
        logging.error("Interrupted: %s", str(e))
        cmd = "rm " + mon_dir + "start_inotify_monitor" # remove the monitoring file
        os.system(cmd)
        sys.exit(0)

    


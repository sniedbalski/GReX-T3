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
dir_fil  = "/hdd/data/candidates/T3/candfils/"  # place to save output filterbank files

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


'''# Function to send a slack message (test)
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
'''

def main(path, post=True, test=False):

    # initiate an inotify instance
    i = ia.Inotify()
    # add the directory to monitor to the instance
    i.add_watch(path)

    try:
        # create a test file, as a marker for the start of the monitoring process
        with open(path+'start_inotify_monitor', 'w'): 
            pass
        # loop and monitor the directory
        for event in i.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event

            # print("PATH=[{}] FILENAME=[{}] EVENT_TYPES={}".format(
            #     path, filename, type_names))
            # print(filename)
            
            # if new file is created
            if type_names == ['IN_CREATE']:
                # if the filename ends with .nc
                if filename.endswith('.nc'): # created a new .nc file
                    logging.info(f"New NetCDF file {filename}, waiting to plot.")
                    #print('Created {}'.format(filename))

                    ### T3 goes here. 
                    c = filename.split('.')[0].split('/')[-1].split('-')[-1] # candidate ID
                    filename_json = c+".json"
                    #print('filename = ', filename_json)

                    os.chdir(env_dir)
                    time.sleep(10)

                    try: 
                        v = mon_dir + "grex_dump-"+c+".nc" # voltage file
                        fn_tempfil = dir_plot + "intermediate.fil" # output temporary .fil
                        fn_outfil = dir_fil + "cand{}.fil".format(c) # output dedispersed candidate .fil
                        (cand, tab) = cand_plotter.gen_cand(v, fn_tempfil, fn_outfil, c+'.json')
                        if test==True:
                            logging.info("Running test, no plots made")
                        else:
                            cand_plotter.plot_grex(cand, tab, c+".json") 
                            logging.info("Done with cand_plotter.py")

                            cmd = "rm {}".format(fn_tempfil)
                            print(cmd)
                            os.system(cmd)
                            logging.info("Successfully plotted the canidate!")

                            pdffile = dir_plot + "grex_cand"+filename_json.split('.')[0]+".png"

                            if post==True:
                                try:
                                    upload_to_slack(pdffile) # upload to Slack #candidates channel
                                    logging.info(f"Successfully posted to Slack #candidates!")
                                except Exception as e:
                                    logging.error("Error uploading candidate plot to Slack: %s", str(e))
                                logging.info("DONE")

                            del cand
                    except Exception as e:
                        logging.error("Error plotting candidates: %s", str(e))

    except PermissionError:
        logging.error("Permission denied: Unable to create inotify test file.")

    except Exception as e:
        logging.error("An error occurred: %s", str(e))


if __name__ == '__main__':
    try:
        post = False
        test = True
        main(mon_dir, post=post, test=test)
    except Exception as e:
        print('Interrupted')
        logging.error("Interrupted: %s", str(e))
        cmd = "rm " + mon_dir + "start_inotify_monitor" # remove the monitoring file
        os.system(cmd)
        sys.exit(0)

    


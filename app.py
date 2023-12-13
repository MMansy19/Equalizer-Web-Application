#Import necessary libraries
import streamlit as st
import Equalizer_Functions
import pandas as pd
import soundfile as soundf
import json

# Set page configuration
st.set_page_config(page_title="Equalizer", page_icon="🎚️", layout="wide")
with open("style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)

st.write(" ")
st.write(" ")
st.write(" ")
#____SESSION STATE INITIALIZATIONS___#
dictnoary_values = {}
values_slider =0

if 'Spectogram_Graph' not in st.session_state:
   st.session_state['Spectogram_Graph'] = False 
st.session_state.size1=0
st.session_state.flag=0
st.session_state.i=0
st.session_state.start=0

#___SIDEBAR SECTION_____#

with st.sidebar:
# Add a title to the sidebar
    st.markdown('<h1 class="sidebar-title">Equalizer</h1>', unsafe_allow_html=True)
    # Create a file uploader in the sidebar and accept only .wav and .audio file types
    file = st.file_uploader("", type=["wav","mp3", "audio"], accept_multiple_files=False)
    # Add a title to choose the mode in the sidebar
    st.markdown('<h2 class="sidebar-title">Modes</h2>', unsafe_allow_html=True)
    # Create a drop-down selector for choosing the mode in the sidebar
    Mode = st.selectbox(label="", options=[
        'Uniform Range', 'Vowels', 'Music Instruments', 'Biological Signal Abnormalities'])
    
#_____MAIN CODE_____#
if file:

        magnitude_at_time, sample_rate = Equalizer_Functions.load_audio_file(file)
        maximum_frequency = Equalizer_Functions.Get_Max_Frequency(magnitude_at_time, sample_rate)  #Get Max frequency
        #Data_frame_of_medical = pd.DataFrame()
        spectogram = st.sidebar.checkbox(label="Spectogram")
        if spectogram:
            st.session_state['Spectogram_Graph'] = True
        else:
            st.session_state['Spectogram_Graph'] = False

        # Depending on the selected equalizer mode, set the appropriate dictionary of frequency ranges and slider values
        #10 Equal ranges of frequency which changes dynamically according to the max freq in the input signal    
        if Mode == 'Uniform Range':
            dictnoary_values = {}
            frequency_step = maximum_frequency/10
            for index in range(10):
                start_freq =index*frequency_step
                end_freq = (index+1)*frequency_step
                freq_range = f"{int(start_freq)}:{int(end_freq)} HZ"
                dictnoary_values[freq_range] = [start_freq,end_freq]
                
            values_slider = [[0,10,1]]*len(list(dictnoary_values.keys()))
            
            # For each frequency range, calculate the number of slider steps required and add it to the slider values list

        elif Mode == 'Vowels':
            # Create a list of vowels
            vowels = ['A', 'E', 'I', 'O', 'U']
            # Create an empty dictionary to store vowel ranges
            dictnoary_values = {}
            # Loop through each vowel
            for vowel in vowels:
                # Set the range of frequencies for each vowel based on its letter

                if vowel == 'A':
                    dictnoary_values[vowel] = [500, 1200]
                elif vowel == 'E':
                    dictnoary_values[vowel] = [800, 1500]
                elif vowel == 'I':
                    dictnoary_values[vowel] = [300, 2500]
                elif vowel == 'O':
                    dictnoary_values[vowel] = [500, 2000]
                elif vowel == 'U':
                    dictnoary_values[vowel] = [200, 900]
            # Create a list of slider values for each vowel
            values_slider = [[0, 10, 1]] * len(vowels)

        elif Mode == 'Music Instruments':
            dictnoary_values = {"Drum ": [0, 500],
                                "Flute": [500, 1000],
                                "Key":  [1000, 2000],
                                "Piano": [2000, 5000]
                                }

            values_slider = [[0, 10, 1]]*len(list(dictnoary_values.keys()))


        elif Mode == 'Biological Signal Abnormalities':
    #         '''
    #         persistent split S2 is a condition that arises when the aortic
    #         and pulmonary valves in the heart do not close simultaneously
    #         causing a split in the second heart sound.
    #         Though it is often benign and does not require treatment,
    #         associated heart conditions may require intervention.
    #           It is a type of abnormalities in the heart.
    #    '''
            # '''
            # Normal: The aortic valve closes before the pulmonary valve (with a small split)  

            # Wide splitting: The duration of splitting increases with inispiration. 

            # Fixed splitting: The duration of splitting doesn't change with both inspiration and expiration 

            # Paradoxical splitting: Reverse the normal

            # '''
            abnormality_sliders=[[],[],[],[],[]]
            dictnoary_values = {}
            keys=['Normal_low_1','Normal_low_2','Normal_high_1','Normal_high_2','Paradoxical']
         
            windowed_smoothed_signal = Equalizer_Functions.applying_hanning_window(magnitude_at_time)    
            if file.name  == 'Normal Split Second Sound.wav':
                abnormality_sliders=Equalizer_Functions.Abnormality_sliders(windowed_smoothed_signal,sample_rate)
                Equalizer_Functions.Json(abnormality_sliders)
            count=0
            for slider in range(len(abnormality_sliders)):
                # Set the range of frequencies for each vowel based on its letter
                if slider ==count:
                    dictnoary_values[keys[count]] = abnormality_sliders[count]
                count+=1
              
            # Create a list of slider values for each vowel
            values_slider = [[0, 10, 1]] * len(dictnoary_values.keys())




            # st.write(abnormality_sliders)
        if Mode != 'Biological Signal Abnormalities': 
            windowed_smoothed_signal = Equalizer_Functions.applying_hanning_window(magnitude_at_time)    

        Equalizer_Functions.processing_signal(Mode, list(dictnoary_values.keys()), values_slider, windowed_smoothed_signal,
                                sample_rate, st.session_state['Spectogram_Graph'], dictnoary_values,file)
            

        # col1,col2=st.columns(2)
        # if Mode == 'Biological Signal Abnormalities': 

        #     with col1:
        #         st.image('Images\S2  Heart Image.png', width=500)

        #     with col2:
        #         st.image('Images\Wiggers_Diagram.svg.png')


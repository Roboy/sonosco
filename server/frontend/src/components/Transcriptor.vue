<template>
  <div class="Transcriptor">
    <h3 v-if="isConnected">Ready to transcribe!</h3>

    <div class="md-layout">

      <div class="md-layout">
        <model class="md-layout-item" v-for="model in this.$store.state.models" v-bind:key="model.id" v-bind:model_id="model.id" v-bind:name="model.name"></model>
      </div>

      <div id="popup" v-if="popupVisible" class="popup">
        <md-card>
          <md-card-header>
            <div class="md-title">Help us improve our model!</div>
          </md-card-header>
          <md-card-content>
            Please enter the correct transcription.<br/>
          </md-card-content>
          <md-card-content>
            <md-field>
              <md-textarea v-model="editableTranscript"></md-textarea>
              <span class="md-error">There is an error</span>
            </md-field>
          </md-card-content>
          <div align="center" style="margin-bottom: 2px">
            <md-button class="md-raised md-primary" @click="saveTranscript()">Improve!</md-button>
            <md-button class="md-raised md-accent" @click="cancel()">I don't want to help</md-button>
          </div>
        </md-card>
      </div>

    </div>

    <div class="controls">
      <md-button class="md-raised" @click="recordStop()">{{ recordButtonText }}</md-button>
      <md-button class="md-raised" @click="playAudio()">Play</md-button>
      <md-button class="md-raised" @click="transcribe()">Transcribe</md-button>
    </div>

  </div>
</template>

<script>
import Model from './Model'

let audioChunks = null
let audioBlob = null
const recordAudio = () =>
  new Promise(async resolve => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus' })
    audioChunks = []
    mediaRecorder.addEventListener('dataavailable', event => {
      audioChunks.push(event.data)
    })
    const start = () => mediaRecorder.start()
    const stop = () =>
      new Promise(resolve => {
        mediaRecorder.addEventListener('stop', () => {
          audioBlob = new Blob(audioChunks, { 'type': 'audio/wav;' })
          const audioUrl = URL.createObjectURL(audioBlob)
          const audio = new Audio(audioUrl)
          const play = () => audio.play()
          resolve({ audioBlob, audioUrl, play })
        })
        mediaRecorder.stop()
      })
    resolve({ start, stop })
  })
let recorder = null
let audio = null

export default {
  name: 'Transcriptor',

  components: {
    'model': Model
  },

  data () {
    return {
      userId: '',
      popupVisible: false,
      isConnected: false,
      editableTranscript: '',
      socketMessage: '',
      recordButtonText: 'Record'
    }
  },

  sockets: {
    connect () {
      // Fired when the socket connects.
      this.isConnected = true
    },

    disconnect () {
      this.isConnected = false
    },

    // Fired when the server sends something on the "transcription" channel.
    transcription (data) {
      this.socketMessage = data
      this.popupVisible = true

      let transcriptions = data[this.$store.getters.getPickedModels.map(el => el['id'])[0]]

      if (Array.isArray(transcriptions)) {
        this.editableTranscript = transcriptions[0]
      }
      else {
        this.editableTranscript = transcriptions
      }
    }
  },

  methods: {
    checkCookies () {
      if (this.$cookies.isKey('userID')) {
        this.userId = window.$cookies.get('userID');
      } else {
        this.userId = Math.random().toString(36).substring(2) + Date.now().toString(36);
        this.$cookies.set('userID', String(this.userId), 1000);
      }
    },
    cancel () {
      this.popupVisible = false
      this.editableTranscript = ''
    },
    async saveTranscript () {
      this.checkCookies()
      this.$socket.emit('saveSample', audioBlob, this.editableTranscript, this.userId)
      console.log("Transcript to be saved: ", this.editableTranscript)
      this.popupVisible = false
    },
    async recordStop () {
      if (recorder) {
        audio = await recorder.stop()
        recorder = null
        this.socketMessage = ''
        this.editableTranscript = ''
        this.recordButtonText = 'Record'
        // document.querySelector('#play-audio-button').removeAttribute('disabled')
      } else {
        recorder = await recordAudio()
        recorder.start()
        this.recordButtonText = 'Stop'
      }
    },
    async playAudio () {
      if (audio && typeof audio.play === 'function') {
        audio.play()
      }
    },
    async transcribe () {
      if (audio && typeof audio.play === 'function') {
        this.$socket.emit('transcribe', audioBlob, this.$store.getters.getPickedModels.map(el => el['id']))
      }
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  h1, h2 {
    font-weight: normal;
  }

  ul {
    list-style-type: none;
    padding: 0;
  }

  li {
    display: inline-block;
    margin: 0 10px;
  }

  a {
    color: #42b983;
  }

  .Transcriptor {
    flex: 1;
    flex-direction: column;
  }

  .controls {
    margin: 20px 10px;
  }

  .popup {
    margin: 20px 10px;
    max-width: 30%;
    min-width: 28%;
  }

  .md-layout {
    width: 90%;
  }

  .md-layout-item {
    margin: 5px;
    min-width: 48%;
  }
</style>

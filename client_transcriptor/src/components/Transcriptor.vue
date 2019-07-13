<template>
  <div class="Transcriptor">
    <h3 v-if="isConnected">We're connected to the server!</h3>

    <div class="model-container">
      <model v-for="model in this.$store.state.models" v-bind:key="model.id"  v-bind:model_id="model.id" v-bind:name="model.name"></model>
    </div>

    <div class="controls">
      <md-button class="md-raised" @click="recordStop()">{{ recordButtonText }}</md-button>
      <md-button class="md-raised" @click="playAudio()">Play</md-button>
      <md-button class="md-raised" @click="transcribe()">Transcribe</md-button>
      <md-dialog-prompt
        :md-active.sync="activeName"
        v-model="name"
        md-title="What is your name?"
        md-input-maxlength="30"
        md-input-placeholder="Type your name..."
        md-confirm-text="Save"
        @md-confirm="onSavePressed()""
        @md-cancel="onNameRequestCanceled()"
        />
      <md-dialog-prompt
        :md-active.sync="activeTrans"
        v-model="writtenTranscript"
        md-title="What did you say?"
        md-input-maxlength="30"
        md-input-placeholder="Type what you said..."
        md-confirm-text="Done"
        @md-cancel="onPromptCanceled()"
        />
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
      activeName: false,
      activeTrans: false,
      nameSaved: false,
      transPromptDone: false,
      writtenTranscript: null,
      name: null,
      isConnected: false,
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
    }
  },

  methods: {
    onNameRequestCanceled (){
      this.name = null
    },
    onSavePressed (){
      this.nameSaved = true
      this.activeTrans = true
    },
    onPromptCanceled() {
      console.log("writtenTranscript at canceled: ",this.writtenTranscript)
      console.log("Prompt canceled")
      this.writtenTranscript = null
      this.promptDone = false

    },
    async recordStop () {
      if (recorder) {
        audio = await recorder.stop()
        recorder = null
        this.socketMessage = ''
        if (this.nameSaved == false){
          this.activeName = true
        } else {
          this.activeTrans = true
        }
        console.log("writtenTranscript at stop: ",this.writtenTranscript)
        this.recordButtonText = 'Record'
        this.transPromptDone = true
        // document.querySelector('#play-audio-button').removeAttribute('disabled')
      } else {
        console.log("writtenTranscript at start: ",this.writtenTranscript)
        this.writtenTranscript = null
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
        this.$socket.emit('record', audioBlob, this.$store.getters.getPickedModels.map(el => el['id']))
        if (this.transPromptDone == true){
          console.log("transPromp done, writtenTranscript: ", this.writtenTranscript, " from: ", this.name)
        }
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

  .model-container {
    display: grid;
    grid-template-columns: auto auto;
    width: 90%;
  }

  .controls {
    margin: 20px 10px;
  }
</style>

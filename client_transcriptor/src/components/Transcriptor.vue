<template>
  <div class="Transcriptor">
    <h1>Transcriptor</h1>
    <p v-if="isConnected">We're connected to the server!</p>
    <p>Transcription: "{{ socketMessage }}"</p>
    <button @click="recordStop()">{{ recordButtonText }}</button>
    <button @click="playAudio()">Play</button>
    <button @click="transcribe()">Transcribe</button>
  </div>
</template>

<script>
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
  data () {
    return {
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
    async recordStop () {
      if (recorder) {
        audio = await recorder.stop()
        recorder = null
        this.socketMessage = ''
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
        this.$socket.emit('record', audioBlob)
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
</style>

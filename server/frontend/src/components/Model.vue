<template>
  <md-card md-with-hover>
    <md-card-header>
      <div class="md-title">
        {{ name }}
      </div>
    </md-card-header>
    <md-card-content v-html="socketMessage"></md-card-content>
    <md-card-actions>
        <md-button @click="removeMyself()">Close</md-button>
    </md-card-actions>
  </md-card>
</template>

<script>
export default {
  name: 'Model',
  props: ['name', 'model_id'],
  data () {
    return {
      isConnected: false,
      socketMessage: 'This is a transcription.',
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
      let transcriptions = data[this.model_id]

      if (Array.isArray(transcriptions)) {
        if (transcriptions.length === 1) {
          this.socketMessage = transcriptions[0]
          return
        }

        var output = '<ol>'

        for (var i = 0; i < transcriptions.length; i++) {
          output += '<li>' + transcriptions[i] + '</li>'
        }

        output += '</ol>'

        this.socketMessage = output
      }
      else {
        this.socketMessage = transcriptions
      }
    }
  },

  methods: {

    removeMyself () {
      this.$store.commit('removeModel', this.model_id)
    }

  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  .Model {
    display: flex;
    flex-direction: column;
    border: solid;
    margin: 5px;
    min-height: 200px;
  }

  .Model h1 {
    text-align: center;
  }

  .Model p {
    padding: 1px 10px 1px 10px;
    flex-grow: 1;
    font-size: 15px;
    font-weight: bold;
  }
</style>

<template>
  <div class="ModelSelector">
    <div>
      <md-field class="select">
          <label for="model">ModelSelector</label>
          <md-select id="model" v-model="chosenModel">
            <md-option v-for="model in models" v-bind:value="model.id" v-bind:key="model.id">{{ model.name }}</md-option>
          </md-select>
      </md-field>
    </div>
    <div class="controls">
      <md-button class="md-raised md-primary" @click="add(chosenModel)" >Add</md-button>
      <md-button class="md-raised md-accent" @click="cancel">Cancel</md-button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ModelSelector',
  data () {
    return {
      models: [
        { 'id': 0, 'name': 'Deepspeech' },
        { 'id': 1, 'name': 'Microsoft' },
        { 'id': 2, 'name': 'Google' }
      ],

      chosenModel: { 'id': 0, 'name': 'Deepspeech' }
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
    add (modelId) {
      this.$store.commit('addModel', this.models[modelId])
      this.$router.push('/')
    },

    cancel () {
      this.$router.push('/')
    }
  }
}
</script>

<style scoped>
  .ModelSelector {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .controls {
    margin: 20px 10px;
  }

  .select {
    margin: 0 6px;
    display: inline-flex;
    width: auto;
  }
</style>

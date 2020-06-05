import { Injectable } from '@angular/core';

import { cube, ISolveRequest, host } from './rubiks';
import { HttpService } from './http.service';
import { deepCopy, CubeService } from './cube.service';

@Injectable({
  providedIn: 'root'
})
export class CommonService {

  status = {
    loading: true,
    connectedToServer: false,
    serverCallId: Math.random(),  // Resets on server change to prevent old server calls messing things up
  };
  scrambleDepth = 10;
  cuda: boolean;
  agents: string[];
  progress = 0;
  timeLimit = 5;
  selectedSearcher: number;
  hasSearchedForSolution = false;
  hasSolution: boolean;
  exploredStates: number;
  actionQueue: number[];
  solveLength: number;
  error: string;

  private breakProgress = false;

  constructor(private httpService: HttpService, private cubeService: CubeService) { }

  get prettyActionQueue(): string {
    return this.actionQueue.map(val => this.cubeService.actions[val]).join(" ");
  }

  public getTimeSearchedStyle(): string {
    return `${this.progress*100}%`;
  }

  public async setSelectedHost(host: host) {
    this.status.serverCallId = Math.random();
    this.httpService.selectedHost = host;
    try {
      await this.getInfo();
    } catch {  // If it cannot connect to a server
      window.location.reload();
    }
  }

  public reset() {
    this.cubeService.reset();
    this.hasSearchedForSolution = false;
    this.actionQueue = [];
  }

  public scramble(moves: number) {
    // Scrambles from current state with animation
    const actions = this.cubeService.scramble(moves);
    this.animateStates(actions);
  }

  private async getInfo() {
    const callId = this.status.serverCallId;
    this.status.loading = true;
    const { cuda, agents } = await this.httpService.getInfo();
    if (callId === this.status.serverCallId) {
      this.cuda = cuda;
      this.agents = agents;
      this.selectedSearcher = 0;
      this.status.loading = false;
      this.status.connectedToServer = true;
      this.error = "";
    }
  }

  public async solve() {
    if (this.status.loading) return;
    this.status.loading = true;
    const solveRequest: ISolveRequest = {
      agentIdx: this.selectedSearcher,
      timeLimit: this.timeLimit,
      state: this.cubeService.currentState,
    };
    this.breakProgress = false;
    this.updateSearchProgress(this.timeLimit);
    this.error = "";
    try {
      const { solution, actions, exploredStates } = await this.httpService.solve(solveRequest);
      this.hasSolution = solution;
      this.exploredStates = exploredStates;
      this.actionQueue = actions;
      this.actionQueue.reverse();  // Allows pop to be used instead of shift which changes runtime complexity from O(n**2) to O(n)
      this.solveLength = this.actionQueue.length;
    } catch (e) {
      this.error = "An error occured. It is most likely caused by a timeout or memory overflow. Try a different searcher or a smaller time limit.";
    }
    this.status.loading = false;
    this.breakProgress = true;
    this.hasSearchedForSolution = true;
  }

  private async updateSearchProgress(timeLimit: number) {
    const fps = 60;
    const frameLength = 1 / fps;
    const frames = Math.round(timeLimit * fps);
    for (let i = 0; i < frames; i ++) {
      await new Promise((resolve, reject) => { setTimeout(() => resolve(), frameLength * 1000); });
      this.progress = (i + 1) / frames;
      if (this.breakProgress) break;
    }
    this.progress = 0;
  }

  public getStates(actions: number[]): cube[] {
    const states: cube[] = [];
    let state = this.cubeService.currentState;
    for (let action of actions) {
      state = this.cubeService.rotate(state, ...this.cubeService.getAction(action));
      states.push(state);
    }
    return states;
  }

  public async animateStates(actions: number[] = null) {
    // Actions should go last to first
    // Uses this.actionQueue if actions is null
    this.status.loading = true;
    this.progress = 0;
    const remainingActions = deepCopy(actions) || this.actionQueue;
    const totalActions = remainingActions.length;
    const frameLength = Math.min(1000 / remainingActions.length, 100);
    while (remainingActions.length) {
      const action = remainingActions.pop();
      this.progress = 1 - remainingActions.length / totalActions;
      await new Promise((resolve, reject) => { setTimeout(() => resolve(), frameLength); });
      this.step(action);
    }
    this.status.loading = false;
    this.progress = 0;
  }

  public step(action: number = null) {
    if (action === null) {
      action = this.actionQueue.pop();
    }
    this.cubeService.currentState = this.cubeService.rotate(this.cubeService.currentState, ...this.cubeService.getAction(action));
  }
}
